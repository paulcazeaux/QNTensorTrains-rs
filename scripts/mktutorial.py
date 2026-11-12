#!/usr/bin/env python3
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Render the tutorial steps under `examples/tutorial/` into the Documenter site.

Each `step_NN_<topic>.rs` is a lesson in the deal.II shape: a `//!` header with the sections
`# Introduction`, `# Builds on`, `# Usage`, `# Results`, `# Possibilities for extensions`, then a
program whose `//` commentary is the narrative. This script turns one file into one page,
`docs/src/tutorial/step-NN.md`:

    title (the header's summary line)
    Introduction · Builds on (links) · Usage
    The program        -- the body split at comment boundaries: prose, code fence, prose, ...
    Results            -- the header's Results prose + the recorded output, if one is on file
    Possibilities for extensions

plus `docs/src/tutorial/index.md`, the course index with the dependency graph drawn from every
step's `# Builds on` line.

Recorded output lives in `docs/src/tutorial/results/step-NN.txt`, TRACKED, so the site builds
without running anything. `--capture DIR` (repeatable) refreshes those files from a golden capture
in the `capture.py` layout: for each step the first DIR whose
`index.tsv` says `ok` wins, and the file's first line records where it came from.

    python3 scripts/mktutorial.py                       # regenerate the pages from the sources
    python3 scripts/mktutorial.py --capture <dir> ...   # also refresh results/ from a capture
    python3 scripts/mktutorial.py --check               # exit 1 if any page is stale (the gate)

`--check` regenerates in memory and compares: a step edited without regenerating its page fails,
as does a page listed nowhere in `docs/make.jl`.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "examples" / "tutorial"
OUT = ROOT / "docs" / "src" / "tutorial"
RESULTS = OUT / "results"
MAKE_JL = ROOT / "docs" / "make.jl"

SECTIONS = ["Introduction", "Builds on", "Usage", "Results", "Possibilities for extensions"]
_STEP = re.compile(r"^step_(\d\d)_([a-z0-9_]+)\.rs$")
_RULE = re.compile(r"^// -{2,}\s*(.*?)\s*-{2,}\s*$")
_RULE_BARE = re.compile(r"^//\s*-{4,}\s*$")
_STEP_REF = re.compile(r"\bstep-(\d\d)\b")


# =================================================================================================
# Parsing one step
# =================================================================================================


_ABS_TESTDATA = re.compile(r"/[^\s|]*?/testdata/")
_STRIP_MARK = re.compile(r"\s*// cfg-strip\([^)]*\): line\s*$")


class Step:
    def __init__(self, path: pathlib.Path):
        m = _STEP.match(path.name)
        if not m:
            raise SystemExit(f"{path}: not a step file (expected step_NN_topic.rs)")
        self.path = path
        self.nn = m.group(1)
        self.topic = m.group(2)
        self.target = f"step-{self.nn}"
        self.page = f"step-{self.nn}.md"
        # Release-strip markers (`// cfg-strip(F): begin|end|line`) are tooling, not lesson text:
        # drop the begin/end lines and the trailing marker on a `line` line.
        lines = [
            _STRIP_MARK.sub("", l)
            for l in path.read_text().split("\n")
            if not l.strip().startswith("// cfg-strip")
        ]
        i = 0
        # License block: leading `//` lines before the first `//!`.
        while i < len(lines) and not lines[i].startswith("//!"):
            i += 1
        header = []
        while i < len(lines) and lines[i].startswith("//!"):
            header.append(lines[i][3:].removeprefix(" "))
            i += 1
        self.summary, self.sections = _split_header(header, path)
        self.body = lines[i:]
        self.builds_on = _STEP_REF.findall(self.sections.get("Builds on", ""))

    @property
    def title(self) -> str:
        return self.summary.rstrip(".")


def _split_header(header: list[str], path: pathlib.Path):
    summary = ""
    sections: dict[str, list[str]] = {}
    cur = None
    for line in header:
        if line.startswith("# "):
            cur = line[2:].strip()
            if cur not in SECTIONS:
                raise SystemExit(f"{path}: unknown header section `# {cur}`")
            sections[cur] = []
        elif cur is None:
            # The summary may wrap onto a second line; it ends at the first blank line.
            if line.strip():
                summary = (summary + " " + line.strip()).strip()
            elif summary and "\n" not in summary:
                summary += "\n"
            if line.strip() and summary.endswith("\n"):
                raise SystemExit(f"{path}: prose before `# Introduction`: {line!r}")
        else:
            sections[cur].append(line)
    missing = [s for s in SECTIONS if s not in sections]
    if missing:
        raise SystemExit(f"{path}: header lacks {missing}")
    if not summary:
        raise SystemExit(f"{path}: no summary line")
    return summary.strip(), {k: "\n".join(v).strip("\n") for k, v in sections.items()}


def _is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") and not s.startswith("//!")


def _comment_text(line: str) -> str:
    s = line.strip()
    s = s[3:] if s.startswith("///") else s[2:]
    return s.removeprefix(" ")


def _program(body: list[str]) -> str:
    """The commented program: comment runs become prose, everything else a `rust` fence."""
    out: list[str] = []
    code: list[str] = []
    prose: list[str] = []

    def flush_code():
        while code and not code[0].strip():
            code.pop(0)
        while code and not code[-1].strip():
            code.pop()
        if code:
            out.append("```rust\n" + "\n".join(code) + "\n```")
        code.clear()

    def flush_prose():
        text = "\n".join(prose).strip("\n")
        if text:
            out.append(text)
        prose.clear()

    for line in body:
        if _is_comment(line):
            flush_code()
            s = line.strip()
            if _RULE_BARE.match(s):
                continue
            m = _RULE.match(s)
            if m:
                flush_prose()
                if m.group(1):
                    out.append(f"### {m.group(1)}")
                continue
            prose.append(_comment_text(line))
        else:
            if prose:
                flush_prose()
            code.append(line)
    flush_prose()
    flush_code()
    return "\n\n".join(out)


# =================================================================================================
# Rendering
# =================================================================================================


_RUSTDOC_LINK = re.compile(r"\[(`[^`\]]+`)\](?!\()")


def _link_steps(text: str, self_nn: str | None = None) -> str:
    """Cross-link `step-NN` mentions outside code fences; render rustdoc `[`Item`]` as plain code."""
    out = []
    fenced = False
    for line in text.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
            out.append(line)
            continue
        if not fenced:
            line = _RUSTDOC_LINK.sub(r"\1", line)
            line = _STEP_REF.sub(
                lambda m: m.group(0) if m.group(1) == self_nn else f"[step-{m.group(1)}](step-{m.group(1)}.md)",
                line,
            )
        out.append(line)
    return "\n".join(out)


def render_step(step: Step, steps: dict[str, Step]) -> str:
    s = step.sections
    parts = [f"# {step.target}: {step.title}", ""]
    parts += [_link_steps(s["Introduction"], step.nn), ""]
    parts += ["## Builds on", ""]
    if step.builds_on:
        parts += [
            "\n".join(
                f"- [{steps[n].target}](step-{n}.md): {steps[n].title}" if n in steps else f"- step-{n}"
                for n in step.builds_on
            ),
            "",
        ]
    else:
        parts += [s["Builds on"], ""]
    parts += ["## Usage", "", _link_steps(s["Usage"], step.nn), ""]
    parts += ["## The program", ""]
    parts += [
        f"The full source is [`examples/tutorial/{step.path.name}`]"
        f"(https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/{step.path.name});"
        " the commentary below is the file's own.",
        "",
        _link_steps(_program(step.body), step.nn),
        "",
    ]
    parts += ["## Results", "", _link_steps(s["Results"], step.nn), ""]
    rec = RESULTS / f"{step.target}.txt"
    if rec.exists():
        text = rec.read_text().rstrip("\n")
        first, _, rest = text.partition("\n")
        parts += [
            f"Recorded output ({first.removeprefix('# ').strip()}):",
            "",
            "```text\n" + rest.strip("\n") + "\n```",
            "",
        ]
    else:
        parts += ["*No recorded output is on file for this step.*", ""]
    parts += ["## Possibilities for extensions", "", _link_steps(s["Possibilities for extensions"], step.nn), ""]
    return "\n".join(parts).rstrip("\n") + "\n"


def render_index(steps: dict[str, Step]) -> str:
    rows = ["| step | lesson | builds on |", "|---|---|---|"]
    for n in sorted(steps):
        st = steps[n]
        deps = ", ".join(f"[step-{d}](step-{d}.md)" for d in st.builds_on) or "—"
        rows.append(f"| [step-{n}](step-{n}.md) | {st.title} | {deps} |")
    edges = [f"  s{d} --> s{n}" for n in sorted(steps) for d in steps[n].builds_on]
    nodes = [f'  s{n}["step-{n}"]' for n in sorted(steps)]
    mermaid = "\n".join(["graph LR"] + nodes + edges)
    return "\n".join(
        [
            "# Tutorial",
            "",
            "A numbered course, one self-contained program per step, in the shape of the deal.II",
            "tutorial: each page is a step's own header and commentary, split at comment boundaries,",
            "followed by the output of a recorded run. Read in order, or follow the dependency graph.",
            "",
            "```sh",
            "cargo run --release --example step-01              # every step builds by default",
            "cargo run --release --example step-01 -- --help    # its arguments, defaults and knobs",
            "```",
            "",
            "\n".join(rows),
            "",
            "## Dependency graph",
            "",
            "```@raw html",
            '<pre class="mermaid">',
            mermaid,
            "</pre>",
            '<script type="module">',
            'import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";',
            "mermaid.initialize({ startOnLoad: true });",
            "</script>",
            "```",
            "",
            "The pages are generated from the sources by `scripts/mktutorial.py`; the gate fails when a",
            "step changes and its page does not.",
            "",
        ]
    )


# =================================================================================================
# Results refresh
# =================================================================================================


def refresh_results(steps: dict[str, Step], captures: list[pathlib.Path]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for n, st in sorted(steps.items()):
        for cap in captures:
            idx = cap / "index.tsv"
            raw = cap / "raw" / f"{st.target}.txt"
            if not idx.exists() or not raw.exists():
                continue
            verdict = {
                r.split("\t")[0]: r.split("\t")[1]
                for r in idx.read_text().splitlines()[1:]
                if "\t" in r
            }.get(st.target)
            if verdict != "ok":
                continue
            text = raw.read_text()
            text = text.split("\n--- stderr ---\n")[0]
            text = re.sub(r"\n--- exit: .*\n?$", "\n", text)
            # A fixture path printed absolute names the capturing machine's checkout: keep it
            # relative to the repository.
            text = _ABS_TESTDATA.sub("testdata/", text)
            (RESULTS / f"{st.target}.txt").write_text(f"# capture {cap.name}\n" + text.rstrip("\n") + "\n")
            print(f"results: {st.target} <- {cap.name}")
            break
        else:
            print(f"results: {st.target}: no `ok` capture among {[c.name for c in captures]}")


# =================================================================================================
# Driver
# =================================================================================================


def generate(steps: dict[str, Step]) -> dict[pathlib.Path, str]:
    pages = {OUT / st.page: render_step(st, steps) for st in steps.values()}
    pages[OUT / "index.md"] = render_index(steps)
    return pages


def check(pages: dict[pathlib.Path, str]) -> int:
    stale = [p for p, text in pages.items() if not p.exists() or p.read_text() != text]
    make = MAKE_JL.read_text()
    unlisted = [p for p in pages if f'"tutorial/{p.name}"' not in make]
    for p in stale:
        print(f"mktutorial: stale page {p.relative_to(ROOT)} (regenerate: python3 scripts/mktutorial.py)")
    for p in unlisted:
        print(f"mktutorial: {p.relative_to(ROOT)} is not in docs/make.jl pages")
    if stale or unlisted:
        return 1
    print(f"✓ tutorial pages current ({len(pages)} pages)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture", action="append", default=[], metavar="DIR",
                    help="a capture.py output dir; repeatable, first `ok` verdict per step wins")
    ap.add_argument("--check", action="store_true", help="verify the pages are current; write nothing")
    a = ap.parse_args()

    steps = {st.nn: st for st in (Step(p) for p in sorted(SRC.glob("step_*.rs")))}
    if a.capture:
        refresh_results(steps, [pathlib.Path(c).resolve() for c in a.capture])
    pages = generate(steps)
    if a.check:
        return check(pages)
    OUT.mkdir(parents=True, exist_ok=True)
    for p, text in pages.items():
        p.write_text(text)
    print(f"wrote {len(pages)} pages under {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
