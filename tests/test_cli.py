from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", ["CGS/CGS.py", "Pretrain/pretrain.py"])
def test_help_from_unrelated_directory(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert list(tmp_path.iterdir()) == []
