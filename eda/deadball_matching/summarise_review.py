"""Counts from a hand-filled review_sample.csv: the part of the review that may leave the VM.

After watching each row of review_sample.csv (written by deadball_matching.py) and filling in
its `category` column, this prints and optionally writes how often each category occurs per
unmatched group, broken down by the interval's signals (dead_no_tag) or the tag's Type
(tag_no_dead). Frame numbers, seek times and the free-text `notes` are never read into the
output. Rows with an empty category are counted as "(unlabelled)".

Usage
    python eda/deadball_matching/summarise_review.py ~/eda_out/deadball_matching/review_sample.csv \\
        [--out ~/eda_out/deadball_matching/review_labels_summary.csv]

Output columns: group, by, key, category, count. `by` is "all", "signals" or "tag_type".
"""

import argparse
import collections
import csv
import sys

COLUMNS = ["group", "by", "key", "category", "count"]
UNLABELLED = "(unlabelled)"


def count(rows):
    """Counter of (group, by, key, category) over review rows (dicts)."""
    out = collections.Counter()
    for r in rows:
        group = (r.get("group") or "").strip()
        cat = (r.get("category") or "").strip().lower() or UNLABELLED
        out[(group, "all", "", cat)] += 1
        if group == "dead_no_tag":
            out[(group, "signals", (r.get("signals") or "").strip(), cat)] += 1
        elif group == "tag_no_dead":
            out[(group, "tag_type", (r.get("tag_type") or "").strip(), cat)] += 1
    return out


def table(counter):
    return [list(k) + [v] for k, v in sorted(counter.items())]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("review_csv", help="a review_sample.csv with the category column filled in")
    p.add_argument("--out", help="also write the counts to this CSV")
    args = p.parse_args(argv)
    with open(args.review_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("group", "category") if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit("{} has no {} column".format(args.review_csv, " or ".join(missing)))
        rows = table(count(reader))
    writer = csv.writer(sys.stdout)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    if args.out:
        with open(args.out, "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(COLUMNS)
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
