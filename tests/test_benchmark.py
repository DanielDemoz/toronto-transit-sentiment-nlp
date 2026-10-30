import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_published_results_match_a_fresh_run():
    before = json.loads((ROOT / "docs" / "results.json").read_text())
    subprocess.run([sys.executable, str(ROOT / "src" / "benchmark.py")], check=True, cwd=ROOT, capture_output=True)
    after = json.loads((ROOT / "docs" / "results.json").read_text())
    assert before["methods"] == after["methods"]


def test_every_message_scored():
    r = json.loads((ROOT / "docs" / "results.json").read_text())
    assert len(r["rows"]) == r["n"] == sum(r["label_counts"].values())
