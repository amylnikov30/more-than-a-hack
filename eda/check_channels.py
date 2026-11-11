"""List the channels and data-format specs an ATD (FIFA/EPTS) metadata XML declares.

Reads only the metadata XML -- never the raw tracking -- and reports, in declaration order:
  * the sensor channels (Devices/Sensors/Channels),
  * per DataFormatSpecification: frame range and the ordered registers
    (StringRegister names, player channel groups, ball channel group, anything else),
  * whether any declared name looks like a ball-status or ball-owner channel.

Usage
    # competition match (paths from games.asset_paths(), honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/check_channels.py OUT_DIR
    # any other EPTS metadata file, e.g. Metrica Sample Game 3
    python check_channels.py OUT_DIR --meta /path/to/Sample_Game_3_metadata.xml

Prints a short summary; writes the full ordered list of declared names (metadata only, no
tracking values) to OUT_DIR/channels.txt.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

# Words that would suggest a ball-status / ball-owner channel. Matched case-insensitively against
# channel ids, channel names, register names and provider-parameter names.
STATUS_PATTERNS = [r"status", r"alive", r"dead", r"in.?play", r"state", r"owner", r"possess",
                   r"owning", r"team.?in"]
STATUS_RE = re.compile("|".join(STATUS_PATTERNS), re.IGNORECASE)


def _tag(element):
    """Element tag without any XML namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _iter(root, name):
    return (el for el in root.iter() if _tag(el) == name)


def _child_text(element, name):
    for child in element:
        if _tag(child) == name:
            return (child.text or "").strip()
    return ""


def sensor_channels(root):
    """[(sensor_id, channel_id, channel_name, unit)] in document order."""
    out = []
    for sensor in _iter(root, "Sensor"):
        for channel in _iter(sensor, "Channel"):
            out.append((sensor.get("id", ""), channel.get("id", ""),
                        _child_text(channel, "Name"), _child_text(channel, "Unit")))
    return out


def _flatten(register):
    """Ordered leaf entries of a register tree: (kind, name)."""
    leaves = []
    for el in register:
        tag = _tag(el)
        if tag == "SplitRegister":
            leaves.extend(_flatten(el))
        elif tag == "PlayerChannelRef":
            leaves.append(("player", el.get("playerChannelId", "")))
        elif tag == "BallChannelRef":
            leaves.append(("ball", "ball_" + el.get("channelId", "")))
        elif tag == "StringRegister":
            leaves.append(("string", el.get("name", "")))
        else:
            leaves.append(("other:" + tag, el.get("name") or el.get("id")
                           or el.get("channelId") or ""))
    return leaves


def data_specs(root):
    """[{start, end, separator, blocks: [(summary, [names])]}] in document order.

    A block is one top-level register of the spec: e.g. the frame counter, the player list,
    the ball group.
    """
    specs = []
    for spec in _iter(root, "DataFormatSpecification"):
        blocks = []
        for register in spec:
            leaves = _flatten([register]) if _tag(register) != "SplitRegister" \
                else _flatten(register)
            names = [name for _, name in leaves]
            kinds = sorted({kind for kind, _ in leaves})
            if kinds == ["player"]:
                slots, suffixes = _player_slots(names)
                summary = "players: {} slots x [{}]".format(len(slots), ", ".join(suffixes))
            elif kinds == ["ball"]:
                summary = "ball: [{}]".format(", ".join(names))
            elif kinds == ["string"]:
                summary = "string register: {}".format(", ".join(names))
            else:
                summary = "mixed register ({}): {}".format("/".join(kinds), ", ".join(names))
            blocks.append((summary, names))
        specs.append({"start": spec.get("startFrame"), "end": spec.get("endFrame"),
                      "separator": spec.get("separator"), "blocks": blocks})
    return specs


def _player_slots(names):
    """Distinct slot prefixes in order, and the distinct channel suffixes, from 'slot_suffix'."""
    slots, suffixes = [], []
    for name in names:
        slot, _, suffix = name.rpartition("_")
        slot = slot or name
        if slot not in slots:
            slots.append(slot)
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    return slots, suffixes


def status_candidates(root):
    """Declared names that look like ball status / owner: [(where, name)]."""
    hits = []
    for _, cid, cname, _ in sensor_channels(root):
        for name in (cid, cname):
            if name and STATUS_RE.search(name):
                hits.append(("sensor channel", name))
    for el in _iter(root, "PlayerChannel"):
        if STATUS_RE.search(el.get("id", "")) or STATUS_RE.search(el.get("channelId", "")):
            hits.append(("player channel", el.get("id", "")))
    for spec in data_specs(root):
        for _, names in spec["blocks"]:
            hits.extend(("data spec", n) for n in names if STATUS_RE.search(n))
    for param in _iter(root, "ProviderParameter"):
        name = _child_text(param, "Name")
        if name and STATUS_RE.search(name):
            hits.append(("provider parameter", name))
    # de-duplicate, keep order
    seen, out = set(), []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def report(meta_path):
    """(summary_lines, full_lines) for one metadata file."""
    root = ET.parse(meta_path).getroot()
    summary, full = [], []

    channels = sensor_channels(root)
    summary.append("sensor channels ({}): {}".format(
        len(channels), ", ".join("{}/{} ({})".format(s, c, u or "?") for s, c, _, u in channels)))
    full.append("# sensor channels: sensor_id, channel_id, name, unit")
    full.extend("{}, {}, {}, {}".format(*c) for c in channels)

    specs = data_specs(root)
    summary.append("data format specs: {}".format(len(specs)))
    # Consecutive specs with the same register layout are printed once, to keep stdout short.
    # (Sample files split the match into several specs when the slot list changes.)
    groups = []
    for i, spec in enumerate(specs, 1):
        layout = (spec["separator"], tuple(text for text, _ in spec["blocks"]))
        if groups and groups[-1]["layout"] == layout:
            groups[-1]["last"], groups[-1]["end"] = i, spec["end"]
            groups[-1]["name_sets"].add(tuple(n for _, ns in spec["blocks"] for n in ns))
        else:
            groups.append({"layout": layout, "first": i, "last": i, "start": spec["start"],
                           "end": spec["end"],
                           "name_sets": {tuple(n for _, ns in spec["blocks"] for n in ns)}})
    for g in groups:
        label = "spec {}".format(g["first"]) if g["first"] == g["last"] \
            else "specs {}-{}".format(g["first"], g["last"])
        extra = "" if len(g["name_sets"]) == 1 \
            else "; same layout, {} different slot lists".format(len(g["name_sets"]))
        summary.append("  {}: frames {}..{}, separator '{}'{}".format(
            label, g["start"], g["end"], g["layout"][0], extra))
        summary.extend("    {}. {}".format(j, text) for j, text in enumerate(g["layout"][1], 1))
    for i, spec in enumerate(specs, 1):
        full.append("# spec {}: frames {}..{}".format(i, spec["start"], spec["end"]))
        for j, (text, names) in enumerate(spec["blocks"], 1):
            full.append("## block {}: {}".format(j, text))
            full.extend(names)

    hits = status_candidates(root)
    if hits:
        summary.append("ball-status / ball-owner candidates: " +
                       "; ".join("{} '{}'".format(w, n) for w, n in hits))
    else:
        summary.append("ball-status / ball-owner channel: NONE declared "
                       "(only positions/speeds; dead ball and owner must be inferred)")
    return summary, full


def repo_root(start=None):
    """Nearest ancestor of `start` (default: this script's folder) that holds a .git entry;
    the script's own folder if there is none."""
    here = os.path.dirname(os.path.abspath(start or __file__))
    d = here
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return here
        d = parent


def check_out_dir(out_dir):
    """Exit if out_dir is inside the repository holding this script: outputs never go in-repo."""
    root = os.path.realpath(repo_root())
    target = os.path.realpath(os.path.abspath(out_dir))
    if target == root or target.startswith(root + os.sep):
        sys.exit("refusing to write into {} (inside the repo {}); pass a folder outside it, "
                 "e.g. ~/eda_out/channels".format(target, root))


def default_meta(source):
    try:
        import games
    except ImportError:
        sys.exit("cannot import games.py -- set PYTHONPATH to the team repo root, or pass --meta")
    try:
        return games.require(["atd_meta"], source=source)[0]
    except (IOError, OSError) as err:
        sys.exit(str(err))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", help="folder for channels.txt (created if missing)")
    parser.add_argument("--meta", help="metadata XML; default: games.asset_paths()['atd_meta']")
    parser.add_argument("--source", default="tactical", choices=["tactical", "broadcast"],
                        help="which recording, when --meta is not given")
    args = parser.parse_args(argv)
    check_out_dir(args.out_dir)

    meta = args.meta or default_meta(args.source)
    summary, full = report(meta)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "channels.txt")
    with open(out_path, "w") as handle:
        handle.write("\n".join(summary) + "\n\n" + "\n".join(full) + "\n")

    print("metadata: {}".format(os.path.basename(meta)))
    print("\n".join(summary))
    print("full ordered list written to {}".format(out_path))


if __name__ == "__main__":
    main()
