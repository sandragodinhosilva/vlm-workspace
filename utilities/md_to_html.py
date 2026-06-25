#!/usr/bin/env python3
"""
Convert one or more Markdown files to a self-contained, GitHub-styled .html.

Why: to share a Markdown doc with someone who should see it rendered (as the md
preview looks) without needing a Markdown viewer — the output is a single .html
file with inline CSS, openable in any browser / attachable to an email. Sibling of
md_to_docx.py (same dir, same CLI shape).

Stdlib-only (no pandoc / no python-markdown needed), so it runs in ANY interpreter:
    python3 md_to_html.py file.md                 # -> file.html alongside
    python3 md_to_html.py file.md -o out.html
    python3 md_to_html.py docs/*.md               # batch -> sibling .html each
    python3 md_to_html.py docs/*.md -o out_dir/    # batch -> into out_dir/

Supported Markdown: headings, **bold**/*italic*/_italic_/`code`, [links](url),
fenced ```code``` blocks, > blockquotes, - / * lists, | tables |, --- rules, and
raw <details>/<summary> passthrough. Good enough for reports/plans; not a full
CommonMark implementation.
"""

import argparse
import html
import re
import sys
from pathlib import Path

_CSS = """body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.5;color:#1f2328;max-width:1000px;margin:0 auto;padding:32px}
h1,h2,h3{border-bottom:1px solid #d8dee4;padding-bottom:.3em;margin-top:1.5em}h1{font-size:2em}h2{font-size:1.5em}h3{font-size:1.2em;border:none}h4,h5,h6{margin-top:1.2em}
code{background:#eff1f3;padding:.2em .4em;border-radius:6px;font-size:85%;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#f6f8fa;padding:16px;border-radius:6px;overflow:auto;border:1px solid #d8dee4}pre code{background:none;padding:0;font-size:85%;white-space:pre-wrap;word-break:break-word}
table{border-collapse:collapse;margin:1em 0;display:block;overflow:auto}th,td{border:1px solid #d8dee4;padding:6px 13px}tr:nth-child(2n){background:#f6f8fa}
blockquote{border-left:.25em solid #d0d7de;padding:0 1em;color:#59636e;margin:0 0 1em}hr{border:none;border-top:1px solid #d8dee4;margin:24px 0}
details{margin:1em 0;border:1px solid #d8dee4;border-radius:6px;padding:8px 12px}summary{cursor:pointer;font-weight:600}ul{margin:.3em 0}a{color:#0969da}
details>:not(summary){display:none}details[open]>:not(summary){display:revert}summary::before{content:'\\25B6 ';font-size:.8em}details[open] summary::before{content:'\\25BC '}"""


def _inline(s: str) -> str:
    """Inline markdown -> HTML on already plain (non-code) text."""
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"(?<![\w\\])_([^_]+)_(?![\w])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _table(rows: list) -> str:
    body, first = "", True
    for r in rows:
        if re.match(r"^\s*\|?[\s|:-]+\|?\s*$", r):  # separator row
            continue
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        tag = "th" if first else "td"
        body += "<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>"
        first = False
    return f"<table>{body}</table>"


def md_to_html_body(md: str) -> str:
    # 1) stash fenced code blocks so inline rules never touch them
    blocks = []

    def _stash(m):
        blocks.append(m.group(2))
        return f"\x00BLOCK{len(blocks)-1}\x00"

    md = re.sub(r"```(\w*)\n(.*?)```", _stash, md, flags=re.DOTALL)

    out, tbl = [], []
    for ln in md.split("\n"):
        if ln.strip().startswith("|"):
            tbl.append(ln)
            continue
        if tbl:
            out.append(_table(tbl))
            tbl = []

        if ln.startswith("\x00BLOCK"):
            idx = int(re.search(r"\d+", ln).group())
            out.append(f"<pre><code>{html.escape(blocks[idx])}</code></pre>")
        elif re.match(r"^#{1,6} ", ln):
            lvl = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{lvl}>{_inline(ln[lvl:].strip())}</h{lvl}>")
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.strip().startswith(">"):
            out.append(f"<blockquote>{_inline(ln.strip()[1:].strip())}</blockquote>")
        elif re.match(r"^\s*[-*] ", ln):
            indent = len(ln) - len(ln.lstrip(" "))
            depth = indent // 2  # 2 spaces per nesting level
            out.append(f"\x01LI{depth}\x01{_inline(re.sub(r'^\s*[-*] ', '', ln))}</li>")
        elif "<details" in ln or "</details>" in ln or "<summary" in ln:
            out.append(ln)  # raw HTML passthrough
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{_inline(ln)}</p>")
    if tbl:
        out.append(_table(tbl))

    body = "\n".join(out)

    # Build NESTED <ul> from depth-tagged list items (\x01LI<depth>\x01 ... </li>).
    def _nest_lists(text: str) -> str:
        lines = text.split("\n")
        result, stack = [], []  # stack = open <ul> depths
        for ln in lines:
            m = re.match(r"\x01LI(\d+)\x01(.*)", ln, re.DOTALL)
            if m:
                depth = int(m.group(1))
                while len(stack) > depth + 1:
                    result.append("</ul>"); stack.pop()
                while len(stack) < depth + 1:
                    result.append("<ul>"); stack.append(len(stack))
                result.append(f"<li>{m.group(2)}")
            else:
                while stack:
                    result.append("</ul>"); stack.pop()
                result.append(ln)
        while stack:
            result.append("</ul>"); stack.pop()
        return "\n".join(result)

    return _nest_lists(body)


def convert(src: Path, dst: Path) -> None:
    md = src.read_text(encoding="utf-8")
    title = html.escape(src.stem)
    page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{_CSS}</style></head><body>"
            f"{md_to_html_body(md)}</body></html>")
    dst.write_text(page, encoding="utf-8")
    print(f"  {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="Markdown file(s) to convert")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .html (single input) or output DIR (batch). "
                         "Default: sibling .html next to each input.")
    args = ap.parse_args()

    if args.output and len(args.inputs) > 1 and args.output.suffix.lower() == ".html":
        sys.exit("With multiple inputs, -o must be a directory, not a .html file.")

    for src in args.inputs:
        if not src.exists():
            print(f"  SKIP (not found): {src}", file=sys.stderr)
            continue
        if args.output is None:
            dst = src.with_suffix(".html")
        elif len(args.inputs) == 1 and args.output.suffix.lower() == ".html":
            dst = args.output
        else:  # output is a dir
            args.output.mkdir(parents=True, exist_ok=True)
            dst = args.output / (src.stem + ".html")
        convert(src, dst)


if __name__ == "__main__":
    main()
