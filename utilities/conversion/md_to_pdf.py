#!/usr/bin/env python3
"""Render a Markdown report to PDF with its figures placed inline.

WHY THIS EXISTS. The cluster has no pandoc / wkhtmltopdf / LaTeX / browser, so the usual
routes to a PDF are all unavailable. matplotlib can write a multi-page PDF and PIL can read
the PNGs, which is enough for a reading copy of a report: headings, paragraphs, bullets,
tables and the figures the markdown references, laid out on A4 pages.

It is a READING copy, not typesetting. Inline `code`, links and emphasis are rendered as
plain text; the LaTeX tables file is the artefact for a paper.

    python md_to_pdf.py REPORT.md --out REPORT.pdf
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

A4 = (8.27, 11.69)
L, R, TOP, BOT = 0.085, 0.945, 0.955, 0.055
MONO = {"family": "DejaVu Sans Mono"}


def strip_inline(s: str) -> str:
    """Markdown inline -> plain text. Images/links keep their visible text only."""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = s.replace("⭐", "").replace("⛔", "! ").replace("⚠️", "! ").replace("⏸️", "")
    s = s.replace("—", "--").replace("–", "-").replace("−", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s.strip()


class Page:
    """One A4 page; `y` walks down in figure coordinates."""

    def __init__(self, pdf: PdfPages, num: int):
        self.pdf, self.num = pdf, num
        self.fig = plt.figure(figsize=A4)
        self.y = TOP

    def room(self, h: float) -> bool:
        return self.y - h > BOT

    def text(self, s: str, size: float, weight: str = "normal", indent: float = 0.0,
             colour: str = "#000000", gap_before: float = 0.0, wrap: int = 96) -> None:
        self.y -= gap_before
        line_h = size / 720.0 * 1.55
        for ln in textwrap.wrap(s, wrap) or [""]:
            if not self.room(line_h):
                return
            # Every wrapped line carries the SAME indent: indenting only the first left
            # continuation lines hanging at the page margin, which read as new paragraphs.
            self.fig.text(L + indent, self.y, ln, fontsize=size, fontweight=weight,
                          color=colour, va="top", ha="left")
            self.y -= line_h

    def rule(self, colour: str = "#cccccc") -> None:
        self.y -= 0.004
        self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y], color=colour, lw=0.7,
                                       transform=self.fig.transFigure))
        self.y -= 0.010

    DPI = 300

    def close(self) -> None:
        self.fig.text(0.5, 0.022, str(self.num), fontsize=8, color="#888888", ha="center")
        self.pdf.savefig(self.fig, dpi=self.DPI)
        plt.close(self.fig)


class Doc:
    def __init__(self, pdf: PdfPages, raster: bool = False):
        self.pdf, self.n = pdf, 0
        self.raster = raster
        # page index (0-based) -> [(vector pdf, (x, y, w, h) in figure coords)]
        self.vector: dict[int, list] = {}
        self.page = self._new()

    def _new(self) -> Page:
        self.n += 1
        return Page(self.pdf, self.n)

    def brk(self) -> None:
        self.page.close()
        self.page = self._new()

    def need(self, h: float) -> None:
        if not self.page.room(h):
            self.brk()

    # -- blocks -------------------------------------------------------------
    def heading(self, level: int, s: str) -> None:
        size = {1: 17, 2: 14.5, 3: 11.8}.get(level, 10.5)
        self.need(0.075 if level <= 2 else 0.05)
        # Wrap width is measured in CHARACTERS, so it must scale with point size or a big
        # heading overruns the right margin. 9.6pt body fits ~96 chars across the text block.
        self.page.text(s, size, "bold", gap_before=0.030 if level > 1 else 0.018,
                       wrap=max(30, int(96 * 9.6 / size)))
        if level <= 2:
            self.page.rule()
        self.page.y -= 0.006

    def para(self, s: str) -> None:
        self.need(0.04)
        self.page.text(s, 9.6, gap_before=0.010)

    def bullet(self, s: str, depth: int = 0) -> None:
        self.need(0.028)
        ind = 0.022 + depth * 0.026
        self.page.y -= 0.006
        y0 = self.page.y
        self.page.fig.text(L + ind - 0.012, y0, "•", fontsize=9.6, va="top")
        self.page.text(s, 9.6, indent=ind, wrap=92 - depth * 6)

    def numbered(self, num: str, s: str, depth: int = 0) -> None:
        """Ordered item with a hanging indent, so the marker never orphans on its own line."""
        self.need(0.028)
        ind = 0.030 + depth * 0.026
        self.page.y -= 0.006
        self.page.fig.text(L + ind - 0.020, self.page.y, f"{num}.", fontsize=9.6, va="top")
        self.page.text(s, 9.6, indent=ind, wrap=90 - depth * 6)

    def quote(self, lines: list[str]) -> None:
        self.need(0.05)
        self.page.y -= 0.008
        y0 = self.page.y
        for ln in lines:
            self.page.text(ln, 9.0, indent=0.020, colour="#444444", wrap=92)
        self.page.fig.add_artist(plt.Line2D([L + 0.006, L + 0.006], [y0, self.page.y],
                                            color="#bbbbbb", lw=2.0,
                                            transform=self.page.fig.transFigure))

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        widths = [max(len(c) for c in col) for col in zip(*rows)]
        # The budget is DERIVED from the page, not guessed: how many monospace characters
        # actually fit between the margins at this point size. A guessed constant is what
        # let the widest table run off the right edge.
        size = 8.4
        gap = 2
        def fits(w_list, pt):
            adv = pt / 72.0 * 0.602 / A4[0]
            return (sum(w_list) + gap * (ncol - 1)) * adv <= (R - L)
        while not fits(widths, size) and max(widths) > 6:
            widths[widths.index(max(widths))] -= 1
        if not fits(widths, size):              # still too wide: shrink the type instead
            while size > 6.0 and not fits(widths, size):
                size -= 0.2
        widths = [max(w, 4) for w in widths]
        cells = [[textwrap.wrap(c, w) or [""] for c, w in zip(r, widths)] for r in rows]
        hs = [max(len(c) for c in r) for r in cells]
        line_h = size / 720.0 * 1.5
        self.need(min(0.30, (sum(hs[:3]) + 2) * line_h))
        self.page.y -= 0.012
        for i, (r, h) in enumerate(zip(cells, hs)):
            if not self.page.room((h + 1) * line_h):
                self.brk()
                self.page.y -= 0.010
            for k in range(h):
                x = L
                for c, w in zip(r, widths):
                    self.page.fig.text(x, self.page.y, c[k] if k < len(c) else "",
                                       fontsize=size, va="top", ha="left",
                                       fontweight="bold" if i == 0 else "normal", **MONO)
                    # DejaVu Sans Mono advance in figure coords: pt/72 * 0.602 (the mono
                    # advance ratio) / page width. Measured, not guessed.
                    x += (w + gap) * (size / 72.0 * 0.602 / A4[0])
                self.page.y -= line_h
            if i == 0:
                self.page.rule("#888888")
            else:
                self.page.y -= 0.002
        self.page.y -= 0.008

    def figure(self, path: Path, caption: str) -> None:
        if not path.exists():
            self.para(f"[missing figure: {path}]")
            return
        twin = path.with_suffix(".pdf")
        if twin.exists() and not self.raster:
            # VECTOR PATH. matplotlib cannot compose an existing PDF into a page, so the
            # figure is RESERVED here -- caption plus blank space on the text page -- and
            # pypdf stamps the vector twin into that rectangle after the text is written.
            # The figure therefore keeps real type, selectable and sharp at any zoom.
            with Image.open(path) as im:                 # aspect from the PNG sibling
                w, h = im.size
            avail_w = R - L
            disp_h = min(avail_w * h / w, (TOP - BOT) - 0.10)
            if not self.page.room(disp_h + 0.06):
                self.brk()
            self.page.y -= 0.012
            self.page.text(caption, 8.6, "bold", colour="#333333", wrap=112)
            self.page.y -= 0.006
            box = (L, self.page.y - disp_h, avail_w, disp_h)
            self.vector.setdefault(self.n - 1, []).append((twin, box))
            self.page.y -= disp_h + 0.016
            return
        with Image.open(path) as im:
            w, h = im.size
        avail_w = R - L
        disp_h = avail_w * h / w
        cap_h = 0.012 * (len(textwrap.wrap(caption, 110)) or 1) + 0.012
        if disp_h + cap_h > TOP - BOT:                       # too tall even alone
            disp_h = (TOP - BOT) - cap_h - 0.02
        if not self.page.room(disp_h + cap_h + 0.02):
            self.brk()
        self.page.y -= 0.012
        self.page.text(caption, 8.6, "bold", colour="#333333", wrap=112)
        self.page.y -= 0.006
        ax = self.page.fig.add_axes([L, self.page.y - disp_h, avail_w, disp_h])
        # interpolation="none" + a high savefig dpi keeps the source pixels: the default
        # resampling softens text-heavy figures, which is most of what these are.
        ax.imshow(Image.open(path), interpolation="none", resample=False)
        ax.axis("off")
        self.page.y -= disp_h + 0.016


FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def stamp_vectors(base: Path, placements: dict, out: Path) -> None:
    """Draw each vector figure into the rectangle reserved for it on its text page."""
    from pypdf import PdfReader, PdfWriter, Transformation

    reader = PdfReader(str(base))
    writer = PdfWriter()
    for idx, page in enumerate(reader.pages):
        pw, ph = float(page.mediabox.width), float(page.mediabox.height)
        for twin, (x, y, w, h) in placements.get(idx, []):
            fig = PdfReader(str(twin)).pages[0]
            fw, fh = float(fig.mediabox.width), float(fig.mediabox.height)
            # Fit inside the reserved box, preserving the figure's own aspect ratio.
            sc = min(pw * w / fw, ph * h / fh)
            tx = pw * x + (pw * w - fw * sc) / 2.0
            ty = ph * y + (ph * h - fh * sc) / 2.0
            page.merge_transformed_page(
                fig, Transformation().scale(sc, sc).translate(tx, ty))
        writer.add_page(page)
    with out.open("wb") as fh:
        writer.write(fh)


def render(md: Path, out: Path, raster: bool = False) -> None:
    lines = md.read_text().split("\n")
    tmp = out.with_suffix(".textonly.pdf")
    with PdfPages(tmp) as pdf:
        doc = Doc(pdf, raster=raster)
        i, buf, quote_buf, tbl = 0, [], [], []

        def flush_para():
            nonlocal buf
            if buf:
                doc.para(strip_inline(" ".join(buf)))
                buf = []

        def flush_quote():
            nonlocal quote_buf
            if quote_buf:
                doc.quote([strip_inline(q) for q in quote_buf if strip_inline(q)])
                quote_buf = []

        def flush_table():
            nonlocal tbl
            if tbl:
                doc.table(tbl)
                tbl = []

        while i < len(lines):
            raw = lines[i]
            s = raw.rstrip()

            if s.startswith(">"):                              # blockquote: figures live here
                body = s.lstrip(">").strip()
                m = FIG_RE.search(body)
                if m:
                    flush_para()
                    cap = " ".join(strip_inline(q) for q in quote_buf)
                    quote_buf = []
                    doc.figure(Path(m.group(1)), cap)
                else:
                    flush_para()
                    quote_buf.append(body)
                i += 1
                continue
            flush_quote()

            if s.startswith("|"):                              # table
                flush_para()
                cells = [c.strip() for c in s.strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    tbl.append([strip_inline(c) for c in cells])
                i += 1
                continue
            flush_table()

            if not s.strip():
                flush_para()
            elif s.startswith("#"):
                flush_para()
                lv = len(s) - len(s.lstrip("#"))
                doc.heading(lv, strip_inline(s.lstrip("#").strip()))
            elif s.strip().startswith("---") and set(s.strip()) <= {"-"}:
                flush_para()
            elif re.match(r"^\s*[-*]\s+", s):
                flush_para()
                depth = (len(s) - len(s.lstrip())) // 2
                doc.bullet(strip_inline(re.sub(r"^\s*[-*]\s+", "", s)), depth)
            elif re.match(r"^\s*\d+\.\s+", s):
                flush_para()
                num = re.match(r"^\s*(\d+)\.", s).group(1)
                doc.numbered(num, strip_inline(re.sub(r"^\s*\d+\.\s+", "", s)),
                             (len(s) - len(s.lstrip())) // 2)
            else:
                buf.append(s.strip())
            i += 1

        flush_para()
        flush_quote()
        flush_table()
        doc.page.close()
        placements = doc.vector
    if placements:
        stamp_vectors(tmp, placements, out)
        tmp.unlink()
    else:
        tmp.replace(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=300,
                    help="page raster resolution; figures are vector when a twin exists")
    ap.add_argument("--raster", action="store_true",
                    help="embed the PNGs instead of the vector PDF twins")
    a = ap.parse_args()
    Page.DPI = a.dpi
    render(a.markdown, a.out, raster=a.raster)
    print(a.out)
