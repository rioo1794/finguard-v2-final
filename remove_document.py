"""
FinGuard AI - Remove every clause that came from one source document.

Usage (project root):
  python remove_document.py "Maniac Rao vs State of Telangana"

Cleans these files (whichever exist):
  datasets/*_clauses.csv                         raw extraction output
  datasets/clean/*_CLEAN.csv                     output of clean_csvs.py
  datasets/labeled/*_AUTO_LABELED.csv            output of auto_label_clauses.py

The match is on the exact `source_document` name (not case-sensitive).
A backup of every file it changes goes to datasets/backup/.
"""
import csv
import glob
import os
import shutil
import sys

csv.field_size_limit(2**31 - 1)

DOC = sys.argv[1].strip() if len(sys.argv) > 1 else "Maniac Rao vs State of Telangana"
BACKUP_DIR = os.path.join("datasets", "backup")

PATTERNS = [
    os.path.join("datasets", "*_clauses.csv"),
    os.path.join("datasets", "clean", "*_CLEAN.csv"),
    os.path.join("datasets", "labeled", "*_AUTO_LABELED.csv"),
]

files = sorted({p for pat in PATTERNS for p in glob.glob(pat)})
if not files:
    sys.exit("No CSV files found. Run this from the project root (the folder containing datasets/).")

total_removed = 0
all_docs = set()

for path in files:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or "source_document" not in rows[0]:
        print(f"skip {path}: no source_document column")
        continue

    header, body = rows[0], rows[1:]
    col = header.index("source_document")
    all_docs.update(r[col] for r in body if len(r) > col)

    keep = [r for r in body if not (len(r) > col and r[col].strip().lower() == DOC.lower())]
    removed = len(body) - len(keep)
    if removed == 0:
        print(f"{path}: nothing to remove")
        continue

    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_name = path.replace(os.sep, "_").replace("/", "_")
    shutil.copy2(path, os.path.join(BACKUP_DIR, backup_name))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(keep)

    total_removed += removed
    print(f"{path}: removed {removed} rows ({len(body)} -> {len(keep)})")

print(f"\nTotal rows removed: {total_removed}")
if total_removed == 0:
    close = sorted(d for d in all_docs if DOC.lower() in d.lower() or d.lower() in DOC.lower())
    print(f'No exact match for "{DOC}".')
    print("Similar names found:", close if close else "none")
else:
    print(f"Backups saved in {BACKUP_DIR}/")