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
import os
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

class EnvVarSubstitutionRule(Rule):
    """
    Replace occurrences of $::VAR with the value of environment variable VAR.

    - Matches patterns like:
        $::MY_VAR
        $::MY_VAR/and/a/path   -> only the $::MY_VAR part is replaced
    - Does NOT match escaped \\$::VAR (one slash plus $) keeps the backslash + token
    - If the env var is missing, emits a warning and leaves $::VAR as-is
    """

    name = "env_var_substitution_v2"

    # Explanation of the regex:
    # (?<!\\)        - don't match if preceded by a backslash (allow escaping)
    # \$::           - literal marker
    # ([A-Za-z_][A-Za-z0-9_]*) - capture the variable name (letters, digits, underscore)
    # (?=$|[^A-Za-z0-9_]) - ensure the next char is either end-of-string or a non-identifier character
    # Match one or more "word" characters (letters, digits, underscore)
    _pattern = re.compile(r"(?<!\\)\$::(\w+)(?=$|[^A-Za-z0-9_])")

    def apply(self, line: str, meta: dict) -> str:
        # Fast path
        if "$::" not in line:
            return line

        lineno = meta.get("lineno", "?")

        def repl(m: Match) -> str:
            varname = m.group(1)
            value = os.environ.get(varname)
            if value is None:
                # Warn and leave the original token untouched
                sys.stderr.write(
                    f"[WARN] Line {lineno}: environment variable '{varname}' not found; "
                    f"leaving '$::{varname}' unchanged.\n"
                )
                return m.group(0)
            return value

        return self._pattern.sub(repl, line)

class IncludeDirectiveRule(Rule):
    """
    Handle ".include <path> ..." directives.

    Rules:
      - If the line (after leading whitespace) starts with ".include" (case-insensitive),
        take the next whitespace-delimited token as the path.
      - Support quoted paths: .include "/absolute/path" or .include '/absolute/path'
      - If the path is not absolute -> raise ValueError (error).
      - Read the file at the absolute path and return its content (inject in place).
      - Ignore any text after the path token.
      - If file can't be read -> raise IOError (propagates to caller).
      - If the line isn't an include directive -> return it unchanged.
    """

    name = "include_directive"

    # Match leading whitespace, then ".include" token, then whitespace, then the path token.
    # We capture leading whitespace (group 1) and the raw path token (group 2).
    # The path token is the first non-whitespace sequence; it may include quotes.
    _regex = re.compile(r"^(\s*)\.include\b\s+(\S+)", flags=re.IGNORECASE)

    def apply(self, line: str, meta: Dict) -> str:
        # Fast path: if no ".include" text at all, return quickly
        if ".include" not in line.lower():
            return line

        lineno = meta.get("lineno", "?")

        m = self._regex.match(line)
        if not m:
            # Not a starting include directive (or not well-formed) => return unchanged.
            return line

        # Extract the path token
        raw_path_token = m.group(2)

        # Strip quotes if present (single or double)
        if (raw_path_token.startswith('"') and raw_path_token.endswith('"')) or \
           (raw_path_token.startswith("'") and raw_path_token.endswith("'")):
            path = raw_path_token[1:-1]
        else:
            path = raw_path_token

        # Make sure we only consider the path token and ignore everything else on the line.
        # Validate absolute path
        if not os.path.isabs(path):
            raise ValueError(f"Line {lineno}: .include path '{path}' is not absolute; refusing to include relative path.")

        # Read file contents
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as e:
            # Propagate with contextual information
            raise IOError(f"Line {lineno}: failed to read include file '{path}': {e}") from e

        # Ensure content ends with a newline so injection does not accidentally merge lines
        if not content.endswith("\n"):
            content = content + "\n"

        # Return the file contents in place of the include line.
        # We do not re-indent or modify the included content; it's injected as-is.
        return content

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

class SpicePrefixSuperflousRemovalRule(Rule):
    """
    Remove leading SPICE 'X' prefix only if it is NOT immediately followed by a digit.
    Examples:
        "  Xfoo bar"  -> "  foo bar"
        "  X_1 bar"   -> "  _1 bar"
        "  X42 foo"   -> unchanged  (because 'X' is followed by a digit)
        "  Ysomething" -> unchanged
    """

    name = "spiceprefix_conditional_removal"

    def apply(self, line: str, meta: Dict) -> str:
        # Leave blank lines or pure comment lines untouched
        if not line.strip():
            return line

        # Leading indentation
        m = re.match(r"^(\s*)", line)
        leading_ws = m.group(1) if m else ""
        rest = line[len(leading_ws):]

        # Only act if the rest starts with 'X'
        if not rest.startswith("X"):
            return line

        # If the next character exists and is a digit, do nothing
        if len(rest) > 1 and rest[1].isdigit():
            return line

        # Otherwise remove the X
        return f"{leading_ws}{rest[1:]}"

import re
from typing import Dict

class ConditionalLeadingXReplaceRule(Rule):
    """
    Replace a leading 'X' (first non-whitespace char) with a provided replacement string
    when a given substring is found somewhere in the line.

    Parameters
    ----------
    match_substring : str
        The substring to look for in the line. If found (according to case sensitivity),
        the rule will perform the leading-'X' replacement.
    replacement : str
        The string that will replace the leading 'X'. If the line's first non-whitespace
        character isn't 'X', the line is left unchanged.
    case_sensitive : bool
        Whether the substring match should be case-sensitive. Default: False.

    Behavior
    --------
    - If the first non-whitespace character is 'X' and `match_substring` appears in the line,
      the leading 'X' is replaced by `replacement` (preserving leading whitespace).
    - Everything after the replaced 'X' is left intact.
    - If `match_substring` is not found, the line is returned unchanged.
    """
    name = "conditional_leading_x_replace"

    def __init__(self, match_substring: str, replacement: str, case_sensitive: bool = False):
        if not match_substring:
            raise ValueError("match_substring must be a non-empty string")
        self.match_substring = match_substring
        self.replacement = replacement
        self.case_sensitive = case_sensitive
        # precompile a pattern for fast presence checking (we'll use search, not full match)
        flags = 0 if case_sensitive else re.IGNORECASE
        # Escape substring so it is treated literally
        self._match_rx = re.compile(re.escape(match_substring), flags=flags)

    def apply(self, line: str, meta: Dict) -> str:
        # fast path
        if not line or not line.strip():
            return line

        # find leading whitespace and remainder
        m = re.match(r"^(\s*)(.*)$", line, flags=re.DOTALL)
        leading_ws = m.group(1)
        rest = m.group(2)

        if not rest:
            return line

        # only act if the first non-whitespace char is 'X'
        if not rest.startswith("X"):
            return line

        # check whether the match_substring appears in the line according to case sensitivity
        if not self._match_rx.search(line):
            return line  # substring not present -> no replacement

        # perform the replacement of the first 'X' only
        new_rest = self.replacement + rest[1:]
        return f"{leading_ws}{new_rest}"

class Pipeline:
    """
    Simple rule-by-rule pipeline:
        for rule in rules:
            for line in lines:
                apply rule
    """

    def __init__(self, rules: List[Rule]) -> None:
        self.rules = rules

    def process_stream(self, in_stream, out_stream, verbose: bool = False):
        # Read all lines as a simple list (preserve endings)
        lines = in_stream.readlines()

        for rule in self.rules:
            new_lines = []
            for lineno, line in enumerate(lines, start=1):
                meta = {"lineno": lineno, "rule": rule.name}
                new_line = rule.apply(line, meta)

                if verbose and new_line != line:
                    before = line.rstrip("\n")
                    after = new_line.rstrip("\n")
                    sys.stderr.write(f"[{rule.name}] line {lineno}: '{before}' -> '{after}'\n")

                # Allow rules to inject multiple lines (e.g. from .include)
                if "\n" in new_line:
                    parts = new_line.splitlines(keepends=True)
                    new_lines.extend(parts)
                else:
                    new_lines.append(new_line)

            # Update for the next rule
            lines = new_lines

        # Write final output
        for line in lines:
            out_stream.write(line)

def build_default_pipeline() -> Pipeline:
    """
    Construct the default pipeline in the requested order:
      1) Spice prefix removal
      2) R-parameter renaming
    """
    rules: List[Rule] = [
        EnvVarSubstitutionRule(),
        IncludeDirectiveRule(),
        #SpicePrefixRemovalRule(),
        ConditionalLeadingXReplaceRule("cap_nmos", "C"),
        ConditionalLeadingXReplaceRule("ppolyf_u", "R"),
        ConditionalLeadingXReplaceRule("nfet", "M"),
        ConditionalLeadingXReplaceRule("pfet", "M"),
        SpicePrefixSuperflousRemovalRule(),
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

