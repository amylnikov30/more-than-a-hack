"""Point the EDA notebooks' cell sources at the eda/ layout (scripts/, notebooks/).

    python tools/update_eda_notebook_paths.py [--dry-run] [NOTEBOOK ...]

Default: every eda/notebooks/*.ipynb next to this repo's tools/ folder. Only the `source` of
code and markdown cells changes; outputs, metadata and execution counts are left alone, and
a notebook is rewritten only if a source changed. The notebooks may hold outputs derived from
the competition match, so this prints only file names and counts, never cell content.

Patterns
    script_path   eda/<script>.py -> eda/scripts/<script>.py, for the ten EDA scripts
    jupyter_cwd   on a line that mentions Jupyter, the working directory, plot_helpers,
                  EDA_DIR or cwd: a bare `eda/` (not followed by more path) -> eda/notebooks/

It also counts, without changing them, what may still need a look by hand: bare `eda/` left
on other lines, and code cells that import one of the scripts (they now live in eda/scripts/,
not beside the notebooks).

Byte-for-byte outputs: nbformat reads each notebook and gives the cell sources the patterns
run on. Writing it back with nbformat would reformat the JSON, so the file is instead
re-serialised with the json settings that reproduce its exact original bytes (found by trying
the usual indent / ensure_ascii / trailing-newline variants), with only the changed `source`
values swapped in. If no setting reproduces the original, or the outputs differ afterwards,
the file is left as it is and the script exits non-zero. Re-running is a no-op.
"""

import argparse
import collections
import glob
import itertools
import json
import os
import re
import sys

import nbformat

SCRIPTS = ("ball_gaps", "ball_outside", "ball_stationary", "check_channels", "common",
           "gap_stoppage", "sample_long_gaps", "signal_profiles", "speed_units", "tag_gaps")

SCRIPT_PATH = re.compile(r"(?<![\w.-])eda/(" + "|".join(SCRIPTS) + r")\.py\b")
CWD_LINE = re.compile(r"jupyter|working[ -]director|plot_helpers|EDA_DIR|\bcwd\b",
                      re.IGNORECASE)
BARE_EDA = re.compile(r"(?<![\w.~-])eda/(?![\w.*-])")
SCRIPT_IMPORT = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(SCRIPTS) + r")\b",
                           re.MULTILINE)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fix_source(src, counts):
    src, n = SCRIPT_PATH.subn(r"eda/scripts/\1.py", src)
    counts["script_path"] += n
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if CWD_LINE.search(line):
            lines[i], n = BARE_EDA.subn("eda/notebooks/", line)
            counts["jupyter_cwd"] += n
        else:
            counts["left_bare_eda"] += len(BARE_EDA.findall(line))
    return "\n".join(lines)


def find_format(data, raw):
    """json.dumps keyword set (plus trailing newline) that reproduces `raw` exactly, or None."""
    for indent, ascii_, newline in itertools.product((1, 2, 4, None), (False, True),
                                                     ("\n", "")):
        kw = {"indent": indent, "ensure_ascii": ascii_}
        if indent is not None:
            kw["separators"] = (",", ": ")
        if (json.dumps(data, **kw) + newline).encode("utf-8") == raw:
            return kw, newline
    return None


def as_stored(new, old):
    """New source text in the stored shape of the old one (list of lines or one string)."""
    if isinstance(old, list):
        return new.splitlines(keepends=True)
    return new


def update(path, dry_run):
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8")
    nb = nbformat.reads(text, as_version=nbformat.NO_CONVERT)
    data = json.loads(text, object_pairs_hook=collections.OrderedDict)
    name = os.path.basename(path)
    if len(data["cells"]) != len(nb.cells):
        print("{}: NOT written (cell count mismatch)".format(name))
        return False

    counts = {"script_path": 0, "jupyter_cwd": 0, "left_bare_eda": 0, "script_imports": 0}
    changed = 0
    for cell, stored in zip(nb.cells, data["cells"]):
        if cell.cell_type not in ("code", "markdown"):
            continue
        new = fix_source(cell.source, counts)
        if cell.cell_type == "code":
            counts["script_imports"] += len(SCRIPT_IMPORT.findall(new))
        if new != cell.source:
            stored["source"] = as_stored(new, stored["source"])
            changed += 1

    print("{}: script_path={} jupyter_cwd={} cells_changed={} "
          "(left as is: bare eda/={}, script imports={})"
          .format(name, counts["script_path"], counts["jupyter_cwd"], changed,
                  counts["left_bare_eda"], counts["script_imports"]))
    if not changed:
        return True

    original = json.loads(text, object_pairs_hook=collections.OrderedDict)
    fmt = find_format(original, raw)
    if fmt is None:
        print("{}: NOT written (no json setting reproduces the original bytes)".format(name))
        return False
    kw, newline = fmt
    out = (json.dumps(data, **kw) + newline).encode("utf-8")

    after = json.loads(out.decode("utf-8"))
    outputs_same = all(a.get("outputs") == b.get("outputs") and
                       a.get("execution_count") == b.get("execution_count") and
                       a.get("metadata") == b.get("metadata")
                       for a, b in zip(original["cells"], after["cells"]))
    if not outputs_same or original["metadata"] != after["metadata"]:
        print("{}: NOT written (outputs or metadata would change)".format(name))
        return False
    if not dry_run:
        with open(path, "wb") as fh:
            fh.write(out)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebooks", nargs="*",
                        help="notebooks to update (default: eda/notebooks/*.ipynb)")
    parser.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = parser.parse_args(argv)
    paths = args.notebooks or sorted(glob.glob(os.path.join(REPO, "eda", "notebooks",
                                                            "*.ipynb")))
    if not paths:
        sys.exit("no notebooks found")
    ok = all([update(p, args.dry_run) for p in paths])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
