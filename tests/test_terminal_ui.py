from __future__ import annotations

from io import StringIO
import subprocess
import sys
from pathlib import Path

import reproduce

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g_metrology.terminal_ui import (
    SBC_BAR_WIDTH,
    SBCProgress,
    format_duration,
    format_sbc_progress,
)


def test_sbc_progress_line_has_requested_shape():
    line = format_sbc_progress(20, 200, 12.0)
    bar = line.split()[2]
    assert len(bar) == SBC_BAR_WIDTH
    assert bar == "||||" + "." * (SBC_BAR_WIDTH - 4)
    assert line.startswith("⠼ ")
    assert "10.0%" in line
    assert "elapsed 00:12" in line
    assert "ETA" in line
    assert "\n" not in line


def test_sbc_progress_updates_one_terminal_line():
    stream = StringIO()
    ui = SBCProgress(10, stream=stream, enabled=True, start_time=0.0, auto_start=False)
    ui.update(0)
    ui.update(1)
    ui.update(5)
    ui.update(10)
    text = stream.getvalue()
    assert text.count("\r") == 5
    assert text.endswith("\n")
    assert text.count("\n") == 1


def test_sbc_spinner_animates_between_progress_updates():
    import time

    class TTYBuffer(StringIO):
        def isatty(self):
            return True

    stream = TTYBuffer()
    ui = SBCProgress(100, stream=stream, enabled=True, spinner_interval=0.02)
    ui.update(1)
    time.sleep(0.08)
    ui.finish()
    text = stream.getvalue()
    # Multiple carriage-return frames verify that animation occurs independently of
    # simulation-completion callbacks.
    assert text.count("\r") >= 3
    assert len({line[0] for line in text.split("\r") if line}) >= 2


def test_sbc_progress_is_silent_when_not_interactive():
    stream = StringIO()
    ui = SBCProgress(10, stream=stream, enabled=False)
    ui.update(0)
    ui.update(10)
    assert stream.getvalue() == ""


def test_help_uses_core_full_archive_names_and_sbc_counts():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "reproduce.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    out = " ".join(proc.stdout.lower().split())
    assert "--core" in out and "--full" in out and "--archive" in out
    assert "200-simulation sbc" in out
    assert "1,000-simulation sbc" in out


def test_sbc_dispatch_counts_are_locked_by_mode():
    assert "--sims" in reproduce.FULL_SBC_ARGS
    assert reproduce.FULL_SBC_ARGS[reproduce.FULL_SBC_ARGS.index("--sims") + 1] == "200"
    assert reproduce.FULL_SBC_ARGS[reproduce.FULL_SBC_ARGS.index("--chains") + 1] == "4"
    assert reproduce.FULL_SBC_ARGS[reproduce.FULL_SBC_ARGS.index("--output-name") + 1] == "sbc_200.json"

    assert reproduce.ARCHIVE_SBC_ARGS[reproduce.ARCHIVE_SBC_ARGS.index("--sims") + 1] == "1000"
    assert reproduce.ARCHIVE_SBC_ARGS[reproduce.ARCHIVE_SBC_ARGS.index("--chains") + 1] == "4"
    assert reproduce.ARCHIVE_SBC_ARGS[reproduce.ARCHIVE_SBC_ARGS.index("--output-name") + 1] == "sbc_1000.json"
    assert reproduce.ARCHIVE_SBC_ARGS[reproduce.ARCHIVE_SBC_ARGS.index("--checkpoint-sims") + 1] == "200"
    assert reproduce.ARCHIVE_SBC_ARGS[reproduce.ARCHIVE_SBC_ARGS.index("--checkpoint-output-name") + 1] == "sbc_200.json"


def test_nuts_diagnostics_stage_output_matches_script_destination():
    stage = next(stage for stage in reproduce._stages_for("archive") if stage.label == "NUTS diagnostics")
    assert stage.outputs == ("results/nuts_reference/nuts_diagnostics.json",)


def test_archive_runs_exactly_one_sbc_stage():
    archive_sbc = [stage for stage in reproduce._stages_for("archive") if stage.kind.startswith("sbc")]
    full_sbc = [stage for stage in reproduce._stages_for("full") if stage.kind.startswith("sbc")]
    assert len(full_sbc) == 1
    assert full_sbc[0].kind == "sbc"
    assert len(archive_sbc) == 1
    assert archive_sbc[0].kind == "sbc_archive"


def test_duration_format():
    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "01:01:01"


def test_archive_uses_one_sbc_run_not_full_plus_archive_sbc():
    archive = [stage for stage in reproduce._stages_for("archive") if stage.kind.startswith("sbc")]
    assert len(archive) == 1
    assert archive[0].kind == "sbc_archive"
    assert reproduce._stages_for("archive").count(archive[0]) == 1
