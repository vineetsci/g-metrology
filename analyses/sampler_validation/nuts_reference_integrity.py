#!/usr/bin/env python3
"""Check the presence and declared structure of the saved NUTS reference artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "nuts_reference"
MANIFEST = RESULTS / "run_manifest.json"

EXPECTED = {
    "free": ["mu", "tau", "lambda", "nu_minus2", "nu", "Sigma", "mu_g", "tau_g"],
    "codata": ["tau", "lambda", "nu_minus2", "nu", "Sigma", "mu_g", "tau_g"],
}


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing NUTS reference manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, variables in EXPECTED.items():
        path = RESULTS / f"{name}.nc"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty NUTS reference artifact: {path}")
        entry = manifest.get(name, {})
        if entry.get("file") != f"results/nuts_reference/{name}.nc":
            raise ValueError(f"Unexpected manifest path for {name}")
        if entry.get("summary_keys") != variables:
            raise ValueError(f"Unexpected reference-variable declaration for {name}")
    print("NUTS reference artifacts: present and structurally declared.")


if __name__ == "__main__":
    main()
