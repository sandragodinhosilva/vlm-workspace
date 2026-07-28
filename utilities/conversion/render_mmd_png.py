#!/usr/bin/env python3
"""Render a mermaid .mmd file to PNG (and SVG) offline via playwright + a local mermaid bundle.

No network required: uses the mermaid.min.js already vendored in the vscode-server extension.

Usage:
  render_mmd_png.py <input.mmd> <output.png> [--scale 2] [--width 2400] [--mermaid /path/mermaid.min.js]
"""
import argparse
import base64
import pathlib
import sys

DEFAULT_MERMAID = (
    "/home/sgsilva/.vscode-server/extensions/"
    "arichika.previewseqdiag-vscode-0.8.0/dist/mermaid/mermaid.min.js"
)

HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; background:#ffffff; }
  #wrap { padding:24px; display:inline-block; background:#ffffff; }
  /* Labels are <br/>-authored: let each line keep its own width and DON'T re-wrap,
     otherwise mermaid measures a box narrower than the text it then paints (clipping). */
  .nodeLabel, .nodeLabel p, .edgeLabel, .edgeLabel p { white-space:pre !important; line-height:1.35 !important; }
  /* never let a label be clipped by its own foreignObject box */
  foreignObject { overflow:visible !important; }
  foreignObject > div { overflow:visible !important; height:auto !important; }
</style>
<script>window.__GRAPH_B64 = "{graph_b64}";</script>
<script>{mermaid_js}</script>
</head><body>
<div id="wrap"><pre class="mermaid" id="src">{graph}</pre></div>
<script>
  window.__done = false;
  window.__err  = null;
  // graph text arrives base64-encoded so no HTML/JS escaping layer can touch it
  document.getElementById('src').textContent =
      new TextDecoder().decode(Uint8Array.from(atob(window.__GRAPH_B64), c => c.charCodeAt(0)));
  mermaid.initialize({ startOnLoad:false, maxTextSize:1000000, securityLevel:'loose',
                        flowchart:{ useMaxWidth:false, htmlLabels:true,
                                    nodeSpacing:55, rankSpacing:65, padding:14 } });
  mermaid.run({ querySelector:'.mermaid' })
    .then(() => { window.__done = true; })
    .catch(e => {
        window.__err = (e && (e.message || e.str || e.name)) ? (e.name + ': ' + (e.message || e.str))
                                                             : JSON.stringify(e, Object.getOwnPropertyNames(e||{}));
        window.__done = true;
    });
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor (2 = retina)")
    ap.add_argument("--width", type=int, default=2400, help="viewport width hint")
    ap.add_argument("--mermaid", default=DEFAULT_MERMAID)
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    out = pathlib.Path(args.output)
    mjs = pathlib.Path(args.mermaid)

    for p, what in ((src, "input .mmd"), (mjs, "mermaid bundle")):
        if not p.exists():
            print(f"ERROR: {what} not found: {p}", file=sys.stderr)
            return 2

    graph = src.read_text(encoding="utf-8")
    # NOTE: do NOT str.format() the graph in — the diagram contains literal { } (mermaid
    # %%{init}%% blocks) and HTML entities that format() mangles. Inject via placeholder,
    # and pass the graph as a JS string so the parser sees it byte-for-byte.
    graph_b64 = base64.b64encode(graph.encode("utf-8")).decode("ascii")
    html = (HTML
            .replace("{graph_b64}", graph_b64)
            .replace("{mermaid_js}", mjs.read_text(encoding="utf-8"))
            .replace("{graph}", ""))

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": 1200},
                                device_scale_factor=args.scale)
        page.set_content(html, wait_until="load")
        page.wait_for_function("window.__done === true", timeout=90_000)

        err = page.evaluate("window.__err")
        if err:
            print(f"ERROR: mermaid failed to render:\n{err}", file=sys.stderr)
            browser.close()
            return 3

        # SVG sidecar (vector, for slides / further editing)
        svg = page.evaluate("document.querySelector('#wrap svg').outerHTML")
        out.with_suffix(".svg").write_text(svg, encoding="utf-8")

        el = page.query_selector("#wrap")
        box = el.bounding_box()
        el.screenshot(path=str(out))
        browser.close()

    kb = out.stat().st_size / 1024
    print(f"OK  {out}  ({box['width']:.0f}x{box['height']:.0f} css px "
          f"@ {args.scale}x -> {kb:.0f} KB)")
    print(f"OK  {out.with_suffix('.svg')}  (vector sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
