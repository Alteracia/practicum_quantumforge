#!/usr/bin/env python3
"""
rename_terms.py

Rename proper nouns / terms both inside the text content AND in the filenames
of every Markdown (.md) file in this knowledge base, driven by a JSON mapping.

Mapping file (terms_map.json), located next to this script:
    { "source name": "new name", ... }

Behaviour
---------
* Every pair is applied to the body of each .md file (read/written as UTF-8).
* Every pair is also applied to the file *name* (the stem only; the ".md"
  extension is preserved).
* Pairs are applied longest-source-first, so longer phrases win over shorter
  prefixes, e.g. "Time Lords" is handled before "Time Lord", "Daleks" before
  "Dalek", "The Doctor" before any bare word, etc.
* All matching is LITERAL and case-sensitive, exactly as written in the JSON.
  That is why both "The Doctor" and "the Doctor" are listed separately.

Usage
-----
    python rename_terms.py            # apply the changes for real
    python rename_terms.py --dry-run  # preview only; write / rename nothing
"""

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(WORKSPACE, "terms_map.json")


def load_map(path):
    """Load the JSON mapping and order it longest-source-first.

    Any key starting with "_" is treated as documentation/metadata (e.g.
    "_comment") and is ignored, so it is never used as a search term.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("terms_map.json must contain a JSON object {source: target}")
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    # Longest source first prevents prefix conflicts
    # (e.g. "Time Lords" must be replaced before "Time Lord").
    return dict(sorted(data.items(), key=lambda kv: len(kv[0]), reverse=True))


def apply_terms(text, terms):
    """Apply every source->target pair to `text` (literal, longest first)."""
    for src, dst in terms.items():
        if src != dst:
            text = text.replace(src, dst)
    return text


def main():
    dry_run = "--dry-run" in sys.argv

    if not os.path.isfile(MAP_FILE):
        print("ERROR: mapping file not found: " + MAP_FILE)
        sys.exit(1)

    terms = load_map(MAP_FILE)
    print("Loaded {} term mapping(s) from {}:".format(len(terms), os.path.basename(MAP_FILE)))
    for src, dst in terms.items():
        print("    {!r} -> {!r}".format(src, dst))
    print()

    # Only top-level .md files in the workspace (the source/ folder holds PDFs).
    md_files = sorted(
        f for f in os.listdir(WORKSPACE)
        if f.lower().endswith(".md") and os.path.isfile(os.path.join(WORKSPACE, f))
    )

    if not md_files:
        print("No .md files found in the workspace.")
        return

    edited = 0
    renamed = 0

    for filename in md_files:
        path = os.path.join(WORKSPACE, filename)

        # 1) Replace inside the file content.
        with open(path, "r", encoding="utf-8") as fh:
            original = fh.read()
        updated = apply_terms(original, terms)
        content_changed = updated != original
        if content_changed:
            if not dry_run:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(updated)
            edited += 1

        # 2) Replace inside the filename (stem only, keep the extension).
        stem, ext = os.path.splitext(filename)
        new_filename = apply_terms(stem, terms) + ext
        name_changed = new_filename != filename

        status_parts = []
        if content_changed:
            status_parts.append("content updated")
        if name_changed:
            new_path = os.path.join(WORKSPACE, new_filename)
            if os.path.exists(new_path):
                status_parts.append("SKIP rename (target exists): {} -> {}".format(filename, new_filename))
            else:
                if not dry_run:
                    os.rename(path, new_path)
                renamed += 1
                status_parts.append("rename: {} -> {}".format(filename, new_filename))
        if not status_parts:
            status_parts.append("no change")

        print("    {}: {}".format(filename, "; ".join(status_parts)))

    mode = "DRY-RUN (nothing written)" if dry_run else "DONE"
    print()
    print("{}: {} file content updated, {} file(s) renamed.".format(mode, edited, renamed))


if __name__ == "__main__":
    main()
