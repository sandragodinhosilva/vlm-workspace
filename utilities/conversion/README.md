# conversion/ — document & diagram converters

Small, dependency-light converters for turning working files into shareable artifacts.

| Script | Does |
| --- | --- |
| `render_mmd_png.py` | mermaid `.mmd` → PNG (+ SVG sidecar), fully offline |
| `md_to_docx.py` | Markdown → .docx |
| `md_to_html.py` | Markdown → standalone HTML |

---

## `render_mmd_png.py` — mermaid → PNG/SVG

```bash
/home/sgsilva/vlm-post-training-home-venv/bin/python \
  /home/sgsilva/utilities/conversion/render_mmd_png.py \
  <input.mmd> <output.png> [--scale 2] [--width 2400] [--mermaid /path/mermaid.min.js]
```

Writes `<output>.png` **and** `<output>.svg` (vector sidecar — prefer it for slides; the
PNGs are large and don't scale).

**Example — the VObs-tool workflow schema:**

```bash
/home/sgsilva/vlm-post-training-home-venv/bin/python \
  /home/sgsilva/utilities/conversion/render_mmd_png.py \
  /home/sgsilva/vlm-post-training/visual_obs/workflow_tool_use.mmd \
  /mnt/data/sgsilva/results/visual_obs/diagrams/vobs_tool_workflow.png --scale 2
```

→ 4692×9272 px @ 2×, ~1.5 MB. Also renders live in the pipeline-inspector app's
**App Guidance** tab, so the PNG and the app show the same source.

### How it works (and why not `mmdc`)

There is **no node / `mmdc` / npx on this cluster**, and no outbound network for a CDN
fetch. So instead:

- **playwright + chromium** (headless) renders the page.
- **mermaid comes from a local bundle** already vendored in the vscode-server extension:
  `~/.vscode-server/extensions/arichika.previewseqdiag-vscode-0.8.0/dist/mermaid/mermaid.min.js`
  (v11.16.0). Override with `--mermaid` if that extension ever disappears.

One-time setup (chromium ~114 MB, already done):

```bash
/home/sgsilva/vlm-post-training-home-venv/bin/python -m playwright install chromium
```

### Two traps this script already handles

- **Never `str.format()` the graph into the HTML.** Mermaid sources contain literal `{ }`
  (the `%%{init}%%` theme block) and HTML entities like `&lt;think&gt;` — `format()` mangles
  both and you get a parse error pointing at a `</think>` that isn't in your file. The graph
  is injected **base64-encoded** and decoded in the browser, so the parser sees it
  byte-for-byte.
- **Label clipping.** Labels in these diagrams are authored with explicit `<br/>` breaks. If
  CSS lets them re-wrap, mermaid measures a box narrower than the text it then paints and
  the bottom line is cut off. The script forces `white-space:pre` on `.nodeLabel`/`.edgeLabel`
  and `overflow:visible` on the `foreignObject`. **Always eyeball the PNG** — a clipped
  render still exits 0.

### Conventions

- Diagram outputs → `/mnt/data/sgsilva/results/visual_obs/diagrams/` (or the relevant
  results subdir). Never `/tmp`.
- Log real runs: `clog misc <run_name> -- <cmd>`.
