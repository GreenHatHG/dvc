import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from dvc.exceptions import InvalidArgumentError
from dvc.log import logger
from dvc.scm import Git

from . import locked

if TYPE_CHECKING:
    from dvc.repo import Repo
    from dvc.repo.index import ObjectContainer

logger = logger.getChild(__name__)


def _validate_args(**kwargs):
    not_in_remote = kwargs.pop("not_in_remote", None)
    cloud = kwargs.pop("cloud", None)
    remote = kwargs.pop("remote", None)
    if remote and not (cloud or not_in_remote):
        raise InvalidArgumentError("`--remote` requires `--cloud` or `--not-in-remote`")
    if not_in_remote and cloud:
        raise InvalidArgumentError(
            "`--not-in-remote` and `--cloud` are mutually exclusive"
        )
    if not any(kwargs.values()):
        raise InvalidArgumentError(
            "Either of `-w|--workspace`, `-a|--all-branches`, `-T|--all-tags` "
            "`--all-experiments`, `--all-commits`, `--date` or `--rev` "
            "needs to be set."
        )
    if kwargs.get("num") and not kwargs.get("rev"):
        raise InvalidArgumentError("`--num` can only be used alongside `--rev`")


def _used_obj_ids_not_in_remote(
    remote_odb_to_obj_ids: "ObjectContainer", jobs: Optional[int] = None
):
    used_obj_ids = set()
    remote_oids = set()
    for remote_odb, obj_ids in remote_odb_to_obj_ids.items():
        assert remote_odb
        remote_oids.update(
            remote_odb.list_oids_exists(
                {x.value for x in obj_ids if x.value},
                jobs=jobs,
            )
        )
        used_obj_ids.update(obj_ids)
    return {obj for obj in used_obj_ids if obj.value not in remote_oids}


def _exec_git(root_dir: str, cmd: list[str]) -> str:
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        return subprocess.check_output(
            ["git"] + cmd,
            cwd=root_dir,
            env=env,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return ""


def _get_all_reflog_commit_shas(root_dir: str) -> set[str]:
    """
    Scans the git reflog for recently accessed commits.

    Returns:
        A set of full commit SHAs, e.g., {"a1b2c3d...", "e4f5g6h..."}.
    """
    output = _exec_git(root_dir, ["rev-list", "--reflog"])
    return set(output.split()) if output else set()


def _scan_active_branch_reflogs(root_dir: str) -> tuple[dict[str, set[str]], set[str]]:
    """
    Strategy 1: Scan reflogs of local branches (refs/heads/*).

    Returns:
        - A dict mapping SHA -> Set of branch names.
        - A set of all currently existing branch names (found in reflogs).
    """
    sha_to_branches = defaultdict(set)
    existing_branches = set()

    # Get reflogs for all branches with format: "<SHA> <RefName>"
    # Example output: "a1b2c3d refs/heads/master@{0}"
    cmd = ["log", "-g", "--pretty=format:%H %gD", "--all"]
    output = _exec_git(root_dir, cmd)

    # Regex to extract "master" from "refs/heads/master@{0}"
    ref_pattern = re.compile(r"refs/heads/([^@]+)@")

    for line in output.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        sha, ref_desc = parts[0], parts[1]
        match = ref_pattern.search(ref_desc)

        if match:
            branch_name = match.group(1)
            sha_to_branches[sha].add(branch_name)
            existing_branches.add(branch_name)

    return sha_to_branches, existing_branches


def _recover_deleted_branches(
    root_dir: str, existing_branches: set[str]
) -> dict[str, set[str]]:
    """
    Strategy 2: Scan HEAD reflog to find branches that were checked out
    but no longer exist (deleted branches).

    Args:
        root_dir: Repo root.
        existing_branches: A set of active branch names to exclude.
    """
    recovered_branches = defaultdict(set)

    # Get HEAD reflog with format: "<SHA> <Subject/Message>"
    # We need the chronological order to determine the state *before* a checkout.
    cmd = ["reflog", "show", "HEAD", "--pretty=format:%H %gs"]
    output = _exec_git(root_dir, cmd)
    lines = output.splitlines()

    # Regex to capture "feature-x" from "checkout: moving from feature-x to master"
    checkout_pattern = re.compile(r"checkout: moving from (.+?) to")

    # Check if a name looks like a full Git SHA (detached HEAD)
    sha_pattern = re.compile(r"^[0-9a-f]{40}$")

    for i in range(len(lines) - 1):
        parts = lines[i].split(" ", 1)
        if len(parts) < 2:
            continue

        message = parts[1]
        match = checkout_pattern.search(message)

        if not match:
            continue

        branch_name = match.group(1)

        # Ignore if this branch actually exists (handled by Strategy 1)
        if branch_name in existing_branches:
            continue

        # Ignore if we were just in a detached HEAD state (moving from a SHA)
        if sha_pattern.match(branch_name):
            continue

        # The entry at i+1 represents the state of the repo before the move at i.
        # This is the SHA where branch_name was pointing.
        prev_parts = lines[i + 1].split(" ", 1)
        if prev_parts:
            prev_sha = prev_parts[0]
            recovered_branches[prev_sha].add(f"{branch_name} (deleted)")

    return recovered_branches


def _map_commits_to_historic_branches(root_dir: str) -> dict[str, set[str]]:
    """
    Builds a map of commits to associated branch names using reflog history.
    This includes both active branches and heuristically recovered deleted branches.

    Returns:
        A dict mapping commit SHAs to a set of branch names.
        e.g., {"a1b2c3d...": {"master", "old-feature (deleted)"}}
    """
    # Map active branches based on their specific reflogs
    sha_to_branches, existing_branches = _scan_active_branch_reflogs(root_dir)

    # Map deleted branches based on HEAD history, excluding existing ones
    deleted_map = _recover_deleted_branches(root_dir, existing_branches)

    # Merge results
    for sha, branches in deleted_map.items():
        sha_to_branches[sha].update(branches)

    return sha_to_branches


@dataclass
class HeuristicScanResult:
    """Holds the results of a heuristic Git history scan for GC dry run."""

    commits_to_scan: list[str] = field(default_factory=list)
    active_tips: set[str] = field(default_factory=set)
    branch_map: dict[str, set[str]] = field(default_factory=dict)


def _get_heuristic_commits(repo: "Repo") -> "HeuristicScanResult":
    """Performs a heuristic scan of the git history.

    Instead of a full history scan, this function checks more
    efficient sources for relevant commits:
    1.  Reflog (Recent user activity)
    2.  Tips of all branches and tags (Active development)

    Returns:
        HeuristicScanResult: An object containing the scan results.

            - commits_to_scan (list[str]): A combined list of unique commit SHAs
              from reflogs and active tips.
            - active_tips (set[str]): A set of commit SHAs considered "active
              tips", currently pointed to by a branch, tag, or HEAD.
            - reflog_branch_map (dict[str, set[str]]): A mapping from commit SHAs
              to a set of historic branch names (e.g., {"master",
              "feat-x (deleted)"}).
    """
    if not isinstance(repo.scm, Git):
        return HeuristicScanResult()

    # A. Get Reflog
    reflog_commits = _get_all_reflog_commit_shas(repo.root_dir)
    reflog_branch_map = _map_commits_to_historic_branches(repo.root_dir)

    # B. Get Active Tips
    # Collect the latest commits (tips) from all branches and tags, plus HEAD.
    active_tips = set()

    def _safe_add(func, *args):
        try:
            if ret := func(*args):
                active_tips.add(ret)
        except:
            pass

    # Add the current commit(HEAD) where the user is working.
    _safe_add(repo.scm.get_rev)

    for branch in repo.scm.list_branches():
        _safe_add(repo.scm.get_ref, branch)

    for tag in repo.scm.list_tags():
        _safe_add(repo.scm.get_ref, tag)

    # Combine
    commits_to_scan = list(reflog_commits | active_tips)
    return HeuristicScanResult(commits_to_scan, active_tips, reflog_branch_map)


def _build_oid_details(
    repo: "Repo",
    removed_oids: set[str],
    commits_to_scan: list[str],
    active_tips: set[str],
    reflog_branch_map: dict[str, set[str]],
) -> dict[str, dict[str, list[dict]]]:
    """
    Scans the provided commits to find which files reference the removed OIDs.
    Returns: {oid: {file_path: [commit_info_dict]}}
    """
    from dvc.progress import Tqdm
    from dvc.repo.index import collect_files

    oid_details = defaultdict(lambda: defaultdict(list))
    commit_messages = {}

    # 1. Scan Historical/Active Commits
    if commits_to_scan:
        for commit_sha in Tqdm(
            commits_to_scan, desc="Scanning active history", unit="commit"
        ):
            # Cache message
            if commit_sha not in commit_messages:
                try:
                    c = repo.scm.resolve_commit(commit_sha)
                    commit_messages[commit_sha] = c.message.strip().split("\n")[0]
                except:
                    commit_messages[commit_sha] = "<no message>"

            # Determine Context
            source_type = (
                "reachable" if commit_sha in active_tips else "dangling/reflog"
            )
            branches = []
            if source_type == "reachable":
                branches = repo.scm.get_refs_containing(commit_sha)
            elif commit_sha in reflog_branch_map:
                branches = list(reflog_branch_map[commit_sha])

            # Scan files in this commit
            with repo.switch(commit_sha):
                for _path, index in collect_files(repo, onerror=None):
                    for out in index.outs:
                        if out.hash_info and out.hash_info.value in removed_oids:
                            path = os.path.relpath(out.fs_path, repo.root_dir)
                            oid_details[out.hash_info.value][path].append(
                                {
                                    "sha": commit_sha,
                                    "branches": branches,
                                    "type": source_type,
                                    "msg": commit_messages[commit_sha],
                                }
                            )

    # 2. Scan Workspace (Uncommitted)
    for _path, index in collect_files(repo, onerror=None):
        for out in index.outs:
            if out.hash_info and out.hash_info.value in removed_oids:
                path = os.path.relpath(out.fs_path, repo.root_dir)
                oid_details[out.hash_info.value][path].append(
                    {
                        "sha": "workspace",
                        "branches": ["workspace"],
                        "type": "workspace",
                        "msg": "Uncommitted changes",
                    }
                )

    return oid_details


def _log_dry_run_entry(
    oid: str, odb: "ObjectContainer", details: dict[str, list[dict]]
):
    """Formats and logs a single OID entry for the dry-run report."""
    import json

    from dvc.utils.humanize import naturalsize

    try:
        path = odb.oid_to_path(oid)
        size = odb.fs.size(path)
        is_dir = path.endswith(".dir")
        size_str = naturalsize(size) + (" B" if size < 1024 else "")

        # Log File Header
        if details:
            for fpath, occurrences in details.items():
                logger.info("- %s", fpath)
                logger.info(
                    "  %s (%s)%s", oid, size_str, " (directory)" if is_dir else ""
                )

                # Deduplicate and Log Commits
                seen = set()
                unique = [
                    o
                    for o in occurrences
                    if not (o["sha"] in seen or seen.add(o["sha"]))
                ]

                for i, occ in enumerate(unique):
                    if i >= 5:
                        logger.info(f"  ... and {len(unique) - 5} more")
                        break

                    sha = occ["sha"][:7] if occ["sha"] != "workspace" else "workspace"
                    branch_str = ""
                    if occ["type"] == "reachable" and occ["branches"]:
                        branch_str = f" ({', '.join(occ['branches'])})"
                    elif occ["type"] == "dangling/reflog":
                        branch_str = " (detached/reflog)"
                    elif occ["type"] == "detached":
                        branch_str = " (detached)"

                    logger.info(f"  • {sha}{branch_str}: {occ['msg']}")
        else:
            logger.info("- %s (%s)%s", oid, size_str, " (directory)" if is_dir else "")
            logger.info("  (source: unknown - .dvc file lost/uncommitted)")

        # Log Directory Contents
        if is_dir:
            try:
                for entry in json.loads(odb.fs.read_text(path)):
                    esize = entry.get("size", 0)
                    esize_str = naturalsize(esize) + (" B" if esize < 1024 else "")
                    logger.info("    - %s (%s)", entry.get("relpath", ""), esize_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.info("    (could not parse .dir file)")

    except FileNotFoundError:
        # Simplified missing file handler
        if details:
            for fpath in details:
                logger.info("- %s", fpath)
        logger.info("- %s (file missing)", oid)


@locked
def gc(  # noqa: C901, PLR0912, PLR0913
    self: "Repo",
    all_branches: bool = False,
    cloud: bool = False,
    remote: Optional[str] = None,
    with_deps: bool = False,
    all_tags: bool = False,
    all_commits: bool = False,
    all_experiments: bool = False,
    force: bool = False,
    jobs: Optional[int] = None,
    repos: Optional[list[str]] = None,
    workspace: bool = False,
    commit_date: Optional[str] = None,
    rev: Optional[str] = None,
    num: Optional[int] = None,
    not_in_remote: bool = False,
    dry: bool = False,
    skip_failed: bool = False,
):
    # require `workspace` to be true to come into effect.
    # assume `workspace` to be enabled if any of `all_tags`, `all_commits`,
    # `all_experiments` or `all_branches` are enabled.
    _validate_args(
        workspace=workspace,
        all_tags=all_tags,
        all_commits=all_commits,
        all_branches=all_branches,
        all_experiments=all_experiments,
        commit_date=commit_date,
        rev=rev,
        num=num,
        cloud=cloud,
        not_in_remote=not_in_remote,
    )

    from contextlib import ExitStack

    from dvc.progress import Tqdm
    from dvc.repo import Repo
    from dvc_data.hashfile.db import get_index
    from dvc_data.hashfile.gc import gc as ogc

    if not repos:
        repos = []
    all_repos = [Repo(path) for path in repos]

    odb_to_obj_ids: ObjectContainer = {}
    with ExitStack() as stack:
        for repo in all_repos:
            stack.enter_context(repo.lock)

        for repo in [*all_repos, self]:
            for odb, obj_ids in repo.used_objs(
                all_branches=all_branches,
                with_deps=with_deps,
                all_tags=all_tags,
                all_commits=all_commits,
                all_experiments=all_experiments,
                commit_date=commit_date,
                remote=remote,
                force=force,
                jobs=jobs,
                revs=[rev] if rev else None,
                num=num or 1,
                skip_failed=skip_failed,
            ).items():
                if odb not in odb_to_obj_ids:
                    odb_to_obj_ids[odb] = set()
                odb_to_obj_ids[odb].update(obj_ids)

    if cloud or not_in_remote:
        _merge_remote_obj_ids(self, remote, odb_to_obj_ids)
    if not_in_remote:
        used_obj_ids = _used_obj_ids_not_in_remote(odb_to_obj_ids, jobs=jobs)
    else:
        used_obj_ids = set()
        used_obj_ids.update(*odb_to_obj_ids.values())

    used_oids_str = {oid.value for oid in used_obj_ids if oid.value}
    processed_odb_paths = set()

    for scheme, odb in self.cache.by_scheme():
        if not odb:
            continue

        # Deduplication Check
        if hasattr(odb, "path") and odb.path:
            abs_path = os.path.abspath(odb.path)
            if abs_path in processed_odb_paths:
                continue
            processed_odb_paths.add(abs_path)

        if dry:
            all_oids = set(odb.all())
            removed_oids = all_oids - used_oids_str

            if not removed_oids:
                logger.info("No unused '%s' cache to remove.", scheme)
                continue

            logger.info(
                "Collecting file path and revision info for all known objects..."
            )

            scan_result = _get_heuristic_commits(self)
            oid_details = _build_oid_details(
                self,
                removed_oids,
                scan_result.commits_to_scan,
                scan_result.active_tips,
                scan_result.branch_map,
            )

            logger.info(
                "%d objects from %s cache will be removed.", len(removed_oids), scheme
            )

            for oid in Tqdm(
                sorted(removed_oids), desc=f"Collecting {scheme} cache size", unit="obj"
            ):
                _log_dry_run_entry(oid, odb, oid_details.get(oid))

        else:
            # Real execution
            num_removed = ogc(odb, used_obj_ids, jobs=jobs, dry=False)
            if num_removed:
                logger.info("Removed %d objects from %s cache.", num_removed, scheme)
            else:
                logger.info("No unused '%s' cache to remove.", scheme)

    # 4. Handle Cloud GC
    if cloud:
        for remote_odb, obj_ids in odb_to_obj_ids.items():
            assert remote_odb is not None
            num_removed = ogc(remote_odb, obj_ids, jobs=jobs, dry=dry)
            if num_removed:
                get_index(remote_odb).clear()
                logger.info("Removed %d objects from remote.", num_removed)
            else:
                logger.info("No unused cache to remove from remote.")


def _merge_remote_obj_ids(
    repo: "Repo", remote: Optional[str], used_objs: "ObjectContainer"
):
    # Merge default remote used objects with remote-per-output used objects
    default_obj_ids = used_objs.pop(None, set())
    remote_odb = repo.cloud.get_remote_odb(remote, "gc -c", hash_name="md5")
    if remote_odb not in used_objs:
        used_objs[remote_odb] = set()
    used_objs[remote_odb].update(default_obj_ids)
    legacy_odb = repo.cloud.get_remote_odb(remote, "gc -c", hash_name="md5-dos2unix")
    if legacy_odb not in used_objs:
        used_objs[legacy_odb] = set()
    used_objs[legacy_odb].update(default_obj_ids)
