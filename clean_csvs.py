"""
FinGuard AI - One-time CSV cleaner.
Fixes malformed CSVs (embedded commas/newlines in scraped clause text).
Output: *_CLEAN.csv with the same number of columns on every row.
"""
import csv
import os
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # allow very long clause cells

TEXT_COL_DEFAULT = 2  # clause_text position if header has no 'clause_text'


def clean_csv(input_path: str, output_path: str):
    print(f"Cleaning: {input_path}")

    fixed_rows = []
    merged_count = padded_count = 0

    with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        text_idx = header.index("clause_text") if "clause_text" in header else TEXT_COL_DEFAULT
        print(f"  Header ({n_cols} cols): {header}")

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue  # skip blank lines

            if len(row) == n_cols:
                fixed_rows.append(row)
            elif len(row) > n_cols:
                # Stray commas split clause_text into extra fields: glue them back
                extra = len(row) - n_cols
                text = ",".join(row[text_idx: text_idx + 1 + extra])
                fixed_rows.append(row[:text_idx] + [text] + row[text_idx + 1 + extra:])
                merged_count += 1
            else:
                fixed_rows.append(row + [""] * (n_cols - len(row)))
                padded_count += 1

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(fixed_rows)

    print(f"  Rows: {len(fixed_rows)} | merged: {merged_count} | padded: {padded_count}")
    print(f"  Saved -> {output_path}\n")


def verify(path: str):
    """Confirm every row has the same column count."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        counts = {len(r) for r in csv.reader(f)}
    status = "OK" if len(counts) == 1 else "PROBLEM"
    print(f"  Verify {path}: column counts = {counts} -> {status}")


if __name__ == "__main__":
    DATA_DIR = "datasets"
    CLEAN_DIR = os.path.join(DATA_DIR, "clean")
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for name in ["consumer", "financial", "property", "family"]:
        src = os.path.join(DATA_DIR, f"{name}_clauses.csv")
        out = os.path.join(CLEAN_DIR, f"{name}_clauses_CLEAN.csv")
        if os.path.exists(src):
            clean_csv(src, out)
            verify(out)
        else:
            print(f"Not found: {src}")

    print("\nDone. Next: python auto_label_clauses.py")