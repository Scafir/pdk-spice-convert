#!/usr/bin/env python3
"""
spice_adapter.py

Adapt SPICE textfiles by applying ordered transformation rules.

Usage:
    python spice_adapter.py input.spice -o output.spice
    cat input.spice | python spice_adapter.py - > output.spice
"""

from __future__ import annotations
import re
import sys
import argparse
from typing import List, Dict, Callable


class Rule:
    """
    Base class for transformation rules.
    Subclass and implement `apply` to transform a single line.
    """
    name = "base"

    def apply(self, line: str, meta: Dict) -> str:
        """
        Transform a single line. Return transformed line (or same line if unchanged).
        `meta` is a dictionary that may hold extra state (e.g. filename, line number).
        """
        raise NotImplementedError

class GenericDeviceParamRenameRule(Rule):
    """
    Generic rule to rename parameters for lines that start with a specific device character.

    Parameters
    ----------
    device_char : str
        Single character identifying the device type (e.g. "R", "M", "Q").
        Matching is case-insensitive and compares to the first non-whitespace character.
    param_map : Dict[str, str]
        Mapping from source parameter name -> destination parameter name.
        Matching is case-insensitive and uses word boundaries so partial names won't be replaced.
    flags : int, optional
        Regex flags to use when matching parameter assignments (default re.IGNORECASE).
    """
    name = "generic_device_param_rename"

    def __init__(self, device_char: str, param_map: Dict[str, str], flags: int = re.IGNORECASE):
        if not device_char or len(device_char.strip()) != 1:
            raise ValueError("device_char must be a single non-empty character.")
        self.device_char = device_char.upper()
        self.param_map = dict(param_map)  # copy to avoid mutation surprises
        self.flags = flags
        # compile regexes for each source param for performance
        self._regexes = {
            src: re.compile(rf"\b({re.escape(src)})\b(\s*=\s*)([^ \t\n]+)", flags=self.flags)
            for src in self.param_map.keys()
        }

    def apply(self, line: str, meta: dict) -> str:
        # leave blank lines as-is
        if not line.strip():
            return line

        # get leading whitespace and remainder
        leading_ws_match = re.match(r"^(\s*)", line)
        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
        rest = line[len(leading_ws):]

        if not rest:
            return line

        # check the first non-whitespace character matches device_char
        if rest[0].upper() != self.device_char:
            return line

        new_rest = rest
        # apply every parameter mapping (replace all occurrences)
        for src, dst in self.param_map.items():
            rx = self._regexes.get(src)
            if rx is None:
                # safety: compile on the fly if missing
                rx = re.compile(rf"\b({re.escape(src)})\b(\s*=\s*)([^ \t\n]+)", flags=self.flags)
                self._regexes[src] = rx

            def repl(match: re.Match) -> str:
                eq_and_spacing = match.group(2)
                value = match.group(3)
                return f"{dst}{eq_and_spacing}{value}"

            new_rest = rx.sub(repl, new_rest)

        return f"{leading_ws}{new_rest}"

class SpicePrefixRemovalRule(Rule):
    """
    Rule 1: If a line starts with X (after any leading whitespace), remove that X.
    Preserves indentation around the removed X.
    """
    name = "spiceprefix_removal"

    def apply(self, line: str, meta: Dict) -> str:
        # Leave blank lines and comment lines untouched
        if not line.strip():
            return line

        # Find leading whitespace
        leading_ws_match = re.match(r"^(\s*)", line)
        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
        rest = line[len(leading_ws):]

        if rest.startswith("X"):
            # Remove only the first 'X' character
            new_rest = rest[1:]
            return f"{leading_ws}{new_rest}"
        else:
            return line




class Pipeline:
    """
    Holds an ordered list of rules and applies them to each line.
    """
    def __init__(self, rules: List[Rule]) -> None:
        self.rules = rules

    def process_line(self, line: str, meta: Dict) -> str:
        cur = line
        for rule in self.rules:
            new = rule.apply(cur, meta)
            # If verbose requested, we can track changes in meta (handled externally)
            cur = new
        return cur

    def process_stream(self, in_stream, out_stream, verbose: bool = False):
        meta = {}
        for lineno, raw_line in enumerate(in_stream, start=1):
            meta = {"lineno": lineno}
            new_line = self.process_line(raw_line, meta)
            if verbose and new_line != raw_line:
                sys.stderr.write(f"[line {lineno}] transformed -> {new_line.rstrip()}\n")
            out_stream.write(new_line)


def build_default_pipeline() -> Pipeline:
    """
    Construct the default pipeline in the requested order:
      1) Spice prefix removal
      2) R-parameter renaming
    """
    rules: List[Rule] = [
        SpicePrefixRemovalRule(),  # keep first
        GenericDeviceParamRenameRule("R", {"r_width": "W", "r_length": "L"}),
        # Example: also support MOSFET parameter renames
        GenericDeviceParamRenameRule("D", {"area": "A", "pj": "P"}),
        GenericDeviceParamRenameRule("C", {"c_width": "W", "c_length": "L"}),
    ]
    return Pipeline(rules)


def parse_args():
    p = argparse.ArgumentParser(description="Adapt a SPICE textfile by applying transformation rules.")
    p.add_argument("input", nargs="?", default="-", help="Input file path (default: stdin)")
    p.add_argument("-o", "--output", default="-", help="Output file path (default: stdout)")
    p.add_argument("--dry-run", action="store_true", help="Run transformations and print changes to stderr but do not write output (writes to stdout anyway).")
    p.add_argument("--verbose", action="store_true", help="Verbose. Print transformed lines to stderr.")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = build_default_pipeline()

    # Input stream
    if args.input == "-":
        in_stream = sys.stdin
    else:
        in_stream = open(args.input, "r", encoding="utf-8")

    # Output stream
    if args.output == "-":
        out_stream = sys.stdout
    else:
        out_stream = open(args.output, "w", encoding="utf-8")

    try:
        pipeline.process_stream(in_stream, out_stream, verbose=args.verbose)
    finally:
        if in_stream is not sys.stdin:
            in_stream.close()
        if out_stream is not sys.stdout:
            out_stream.close()


if __name__ == "__main__":
    main()

