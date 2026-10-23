from pathlib import Path
import csv, hashlib, sys
ROOT = Path(__file__).resolve().parents[1]
errors = []
with (ROOT / "DATA_MANIFEST.csv").open(encoding="utf-8", newline="") as h:
    rows = list(csv.DictReader(h))
for row in rows:
    p = ROOT / row["path"]
    if not p.is_file(): errors.append(f"Missing: {row['path']}"); continue
    if p.stat().st_size != int(row["bytes"]): errors.append(f"Size mismatch: {row['path']}")
    if "--sha256" in sys.argv:
        digest = hashlib.sha256()
        with p.open("rb") as h:
            for block in iter(lambda: h.read(8*1024*1024), b""): digest.update(block)
        if digest.hexdigest() != row["sha256"]: errors.append(f"Checksum mismatch: {row['path']}")
if errors:
    print("\n".join(errors)); sys.exit(1)
print(f"Verified {len(rows)} input files" + (" including SHA256" if "--sha256" in sys.argv else " (existence and size)"))
