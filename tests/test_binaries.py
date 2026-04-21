from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from turnmux.runtime.binaries import build_runtime_path, resolve_binary


class BinaryResolutionTests(unittest.TestCase):
    def test_build_runtime_path_includes_common_locations(self) -> None:
        runtime_path = build_runtime_path(extra_entries=["/custom/bin", "/usr/bin"])
        segments = runtime_path.split(os.pathsep)

        self.assertEqual(segments[0], "/custom/bin")
        self.assertIn("/opt/homebrew/bin", segments)
        self.assertEqual(segments.count("/usr/bin"), 1)

    def test_resolve_binary_uses_fallback_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            binary_path = Path(tmp_dir) / "tmux"
            binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary_path.chmod(0o755)

            resolved = resolve_binary("tmux-missing", fallback_paths=[binary_path])

        self.assertEqual(resolved, str(binary_path.resolve(strict=False)))
