from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from turnmux.config import RepoPathValidationError, validate_repo_path


class RepoPathValidationTests(unittest.TestCase):
    def test_validate_repo_path_accepts_directories_within_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            allowed_root = tmp_path / "allowed"
            repo_path = allowed_root / "project"
            repo_path.mkdir(parents=True)

            validated_path = validate_repo_path(repo_path, [allowed_root])

            self.assertEqual(validated_path, repo_path.resolve())

    def test_validate_repo_path_rejects_paths_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            allowed_root = tmp_path / "allowed"
            other_path = tmp_path / "other" / "project"
            allowed_root.mkdir()
            other_path.mkdir(parents=True)

            with self.assertRaises(RepoPathValidationError):
                validate_repo_path(other_path, [allowed_root])
