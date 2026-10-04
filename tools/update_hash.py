# update_hash.py — recompute hashes for the three yaml inputs and update
# data/MANIFEST.json and src/g_metrology/data.py.
#
# Writes only after explicit confirmation. Runs only from the repo root folder.
# DO NOT RUN IT UNLESS A HASH UPDATE IS REQUIRED AFTER DELIBERATELY EDITING INPUT ../data/ files.

from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path.cwd()
DATA = ROOT / "data"
MANIFEST = DATA / "MANIFEST.json"
DATA_PY = ROOT / "src" / "g_metrology" / "data.py"

files = [
    DATA / "codata_2010_g.yaml",
    DATA / "codata_2018_g.yaml",
    DATA / "nist2026_configurations.yaml",
]

if not MANIFEST.is_file() or not DATA_PY.is_file():
    sys.exit(
        "Run this script from the repository root. "
        f"Missing {MANIFEST} or {DATA_PY}."
    )

# --- Compute new hashes -------------------------------------
records = {}
for path in files:
    raw = path.read_bytes()
    records[path.name] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    print(f"{path.name}:")
    print(f"  bytes  = {len(raw)}")
    print(f"  sha256 = {records[path.name]['sha256']}")

# --- Compute proposed updates, without writing --------------
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
manifest_changes = []
for record in manifest["files"]:
    name = record["path"]
    if name in records:
        old = record["sha256"]
        new = records[name]["sha256"]
        if old != new:
            manifest_changes.append((name, old, new))
            record["bytes"] = records[name]["bytes"]
            record["sha256"] = new

data_py_text = DATA_PY.read_text(encoding="utf-8")
data_py_new = data_py_text
data_py_changes = []
for name, rec in records.items():
    pattern = rf"('{re.escape(name)}':\s*)'[0-9a-f]+'"
    match = re.search(pattern, data_py_new)
    if not match:
        sys.exit(f"Could not find hash entry for {name} in {DATA_PY}")
    old = re.search(r"'([0-9a-f]+)'", match.group(0)).group(1)
    new = rec["sha256"]
    if old != new:
        data_py_changes.append((name, old, new))
    data_py_new = re.subn(
        pattern,
        rf"\g<1>'{new}'",
        data_py_new,
    )[0]

# --- Report what would change -------------------------------
print()
if not manifest_changes and not data_py_changes:
    print("No changes. Manifest and data.py already match the yaml files.")
    sys.exit(0)

print("Proposed changes:")
for name, old, new in manifest_changes:
    print(f"  data/MANIFEST.json   {name}")
    print(f"    {old} -> {new}")
for name, old, new in data_py_changes:
    print(f"  src/g_metrology/data.py   {name}")
    print(f"    {old} -> {new}")

# --- Require explicit confirmation --------------------------
print()
print("This will overwrite data/MANIFEST.json and src/g_metrology/data.py.")
print("Type YES (or y) to proceed. Anything else aborts.")
answer = input("Confirm: ").strip()

if answer not in {"y", "Y", "YES", "yes", "Yes"}:
    print("Aborted. No files written.")
    sys.exit(1)

# --- Write --------------------------------------------------
MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
DATA_PY.write_text(data_py_new, encoding="utf-8")

print("\nUpdated:")
print("  data/MANIFEST.json")
print("  src/g_metrology/data.py")