"""Filesystem-bound validation for frozen S1 runtime-lock artifacts."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path


def validate_unredirected_artifact_paths(artifact_paths: Sequence[Path]) -> None:
    """Reject redirects in every existing artifact or ancestor component."""
    checked: set[str] = set()
    for artifact_path in artifact_paths:
        absolute = Path(os.path.abspath(artifact_path))
        components = _path_components(absolute)
        for index, path in enumerate(components):
            serialized = str(path)
            if serialized in checked:
                continue
            checked.add(serialized)
            _validate_physical_name(path)
            if not os.path.lexists(path):
                continue
            if _path_is_redirect(path):
                raise ValueError("Runtime-lock output path contains a redirect.")
            is_artifact = index == len(components) - 1
            if (is_artifact and not path.is_file()) or (
                not is_artifact and not path.is_dir()
            ):
                raise ValueError("Runtime-lock output path has an invalid entry type.")


def _validate_physical_name(path: Path) -> None:
    parent = path.parent
    if not parent.is_dir():
        return
    with os.scandir(parent) as entries:
        matches = [
            entry.name
            for entry in entries
            if entry.name.casefold() == path.name.casefold()
        ]
    if len(matches) > 1:
        raise ValueError("Runtime-lock output path has a case-colliding entry.")
    if matches and matches[0] != path.name:
        raise ValueError("Runtime-lock output path casing does not match disk.")


def validate_exact_lock_tree(
    root: Path,
    expected_relative_files: Sequence[Path],
) -> None:
    """Require one unredirected, case-exact filesystem tree."""
    paths = tuple(expected_relative_files)
    if any(
        (
            not isinstance(relative, Path)
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        )
        for relative in paths
    ):
        raise ValueError("Expected runtime-lock path is not canonical.")
    expected_files = {relative.as_posix() for relative in paths}
    if len(expected_files) != len(paths):
        raise ValueError("Expected runtime-lock paths contain duplicates.")
    expected_directories = {
        parent.as_posix()
        for relative in paths
        for parent in relative.parents
        if parent != Path(".")
    }
    expected_casefold: dict[str, str] = {}
    for expected in expected_files | expected_directories:
        _record_case_exact(expected_casefold, expected)
    if not root.is_dir() or _path_is_redirect(root):
        raise ValueError(
            "Runtime-lock root is missing, redirected, or not a directory."
        )

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    observed_casefold: dict[str, str] = {}
    _collect_tree_entries(
        root,
        Path(),
        observed_files,
        observed_directories,
        observed_casefold,
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("Runtime-lock filesystem tree is not exact.")


def _path_components(path: Path) -> list[Path]:
    components: list[Path] = []
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        components.append(current)
    return components


def _collect_tree_entries(
    directory: Path,
    relative_root: Path,
    files: set[str],
    directories: set[str],
    casefold_index: dict[str, str],
) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            relative = relative_root / entry.name
            serialized = relative.as_posix()
            _record_case_exact(casefold_index, serialized)
            if entry.is_symlink() or _stat_is_reparse(
                entry.stat(follow_symlinks=False)
            ):
                raise ValueError("Runtime-lock tree contains a redirected entry.")
            if entry.is_dir(follow_symlinks=False):
                directories.add(serialized)
                _collect_tree_entries(
                    Path(entry.path),
                    relative,
                    files,
                    directories,
                    casefold_index,
                )
            elif entry.is_file(follow_symlinks=False):
                files.add(serialized)
            else:
                raise ValueError("Runtime-lock tree contains an invalid entry type.")


def _record_case_exact(index: dict[str, str], value: str) -> None:
    folded = value.casefold()
    previous = index.setdefault(folded, value)
    if previous != value:
        raise ValueError("Runtime-lock tree contains a case-colliding entry.")


def _path_is_redirect(path: Path) -> bool:
    return path.is_symlink() or _stat_is_reparse(path.stat(follow_symlinks=False))


def _stat_is_reparse(value: os.stat_result) -> bool:
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(value, "st_file_attributes", 0) & marker)
