"""Signal B: manual evidence boxes -> masked images (evidence vs. similar-area control) + review page.

Nothing here infers where the evidence is. ``init`` writes an empty template; a person fills ``relevant_boxes``
(pixel coords [x0, y0, x1, y1] on the original image) using outputs/signal_b/annotate.html or by hand. ``build``
proposes control boxes of the same size that do not overlap the evidence (saved back as ``control_source:
auto_proposed``), renders masked images and a review page. Only items with ``reviewed: true`` are used.
"""

from __future__ import annotations

import html
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .config import output_dir, resolve
from .data import image_hash, load_image, load_manifest


def _ann_path(cfg) -> Path:
    return resolve(cfg, cfg["signal_b"]["annotation_path"])


def init_annotations(cfg: dict) -> None:
    path = _ann_path(cfg)
    rows = load_manifest(cfg, "diagnostic")[: cfg["signal_b"]["max_items"]]
    if path.exists():
        print(f"{path} exists; not overwriting")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        items = [{"sample_id": r["sample_id"], "image_hash": r["image_hash"], "image_size": r["image_size"],
                  "question": r["question"], "answers": r["answers"], "relevant_boxes": [], "control_boxes": [],
                  "control_source": None, "reviewed": False, "notes": ""} for r in rows]
        path.write_text(json.dumps({"format": "signal_b_boxes_v1",
                                    "instructions": "relevant_boxes: [[x0,y0,x1,y1], ...] in original pixels covering "
                                                    "ALL places the answer can be read from (data label, bar/point, "
                                                    "and any table/legend repeating the value). Set reviewed=true "
                                                    "only after checking outputs/signal_b/review.html.",
                                    "items": items}, indent=1, ensure_ascii=False))
        print(f"wrote template {path} ({len(items)} items)")
    _write_annotator(cfg, rows)


def _overlaps(a, b, margin=0):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def propose_control_boxes(relevant: list[list[int]], size: tuple[int, int], margin: int, seed: int) -> list | None:
    """Same-size boxes placed where they do not touch any evidence box. None if no placement exists."""
    rng = random.Random(seed)
    w_img, h_img = size
    placed = []
    for box in relevant:
        w, h = box[2] - box[0], box[3] - box[1]
        for _ in range(2000):
            x0, y0 = rng.randint(0, max(0, w_img - w)), rng.randint(0, max(0, h_img - h))
            c = [x0, y0, x0 + w, y0 + h]
            if not any(_overlaps(c, r, margin) for r in relevant) and not any(_overlaps(c, p) for p in placed):
                placed.append(c)
                break
        else:
            return None
    return placed


def apply_mask(img: Image.Image, boxes, fill) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=tuple(fill))
    return out


def _valid_box(b, size):
    return (len(b) == 4 and 0 <= b[0] < b[2] <= size[0] and 0 <= b[1] < b[3] <= size[1])


def build(cfg: dict) -> None:
    sb = cfg["signal_b"]
    path = _ann_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m vlmrh masks init` first")
    ann = json.loads(path.read_text())
    rows = {r["sample_id"]: r for r in load_manifest(cfg, "diagnostic")}
    out = output_dir(cfg) / "signal_b"
    (out / "images").mkdir(parents=True, exist_ok=True)
    cards, n_ready = [], 0
    for k, it in enumerate(ann["items"]):
        r = rows[it["sample_id"]]
        if it["image_hash"] != r["image_hash"]:
            raise AssertionError(f"annotation image_hash mismatch for {it['sample_id']}")
        rel = it.get("relevant_boxes") or []
        if not rel:
            continue
        bad = [b for b in rel + (it.get("control_boxes") or []) if not _valid_box(b, r["image_size"])]
        if bad:
            raise ValueError(f"{it['sample_id']}: invalid boxes {bad} for size {r['image_size']}")
        if not it.get("control_boxes"):
            ctrl = propose_control_boxes(rel, tuple(r["image_size"]), sb["control_margin_px"], seed=k)
            if ctrl is None:
                print(f"{it['sample_id']}: no non-overlapping control placement; set control_boxes manually")
                continue
            it["control_boxes"], it["control_source"], it["reviewed"] = ctrl, "auto_proposed", False
        img = load_image(r)
        rel_img = apply_mask(img, rel, sb["fill_color"])
        ctl_img = apply_mask(img, it["control_boxes"], sb["fill_color"])
        boxed = img.copy()
        d = ImageDraw.Draw(boxed)
        for b in rel:
            d.rectangle(b, outline=(220, 0, 0), width=3)
        for b in it["control_boxes"]:
            d.rectangle(b, outline=(0, 90, 220), width=3)
        sid = it["sample_id"]
        for name, im in (("boxes", boxed), ("relevant", rel_img), ("control", ctl_img)):
            im.save(out / "images" / f"{sid}_{name}.png")
        area = lambda bs: sum((b[2] - b[0]) * (b[3] - b[1]) for b in bs)  # noqa: E731
        n_ready += bool(it.get("reviewed"))
        cards.append(
            f"<div class='card'><h3>{html.escape(sid)} — reviewed: <b>{it.get('reviewed')}</b>"
            f" (control: {it.get('control_source')})</h3><p>{html.escape(r['question'])} → "
            f"<code>{html.escape(str(r['answers']))}</code></p><p>masked area relevant={area(rel)} "
            f"control={area(it['control_boxes'])} px</p><div class='row'>"
            + "".join(f"<figure><img src='images/{sid}_{n}.png'><figcaption>{n}</figcaption></figure>"
                      for n in ("boxes", "relevant", "control")) + "</div></div>")
    path.write_text(json.dumps(ann, indent=1, ensure_ascii=False))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Signal B Mask Review</title>
<style>:root{{--bg:#fff;--fg:#222}}@media (prefers-color-scheme: dark){{:root{{--bg:#181818;--fg:#ddd}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui;margin:16px}}.row{{display:flex;gap:8px;flex-wrap:wrap}}
figure{{margin:0;flex:1 1 300px}} img{{width:100%;background:#fff}} .card{{border-bottom:1px solid #8886;padding:8px 0}}
</style></head><body><h1>Signal B mask review</h1>
<p>Red = evidence (relevant) boxes, blue = control boxes. For each item check: (1) the relevant mask removes
<b>every</b> place the answer value can be read (data label, bar/point height readable from the axis, legend/table);
(2) the control mask covers no answer evidence. Then set <code>reviewed: true</code> in
<code>{html.escape(str(path))}</code>. Items ready for evaluation: {n_ready}.</p>{''.join(cards)}</body></html>"""
    (out / "review.html").write_text(page)
    print(f"built {len(cards)} annotated items ({n_ready} reviewed) -> {out}/review.html")


def load_ready_items(cfg: dict, diag_rows: list[dict]) -> tuple[list[dict], str]:
    """Reviewed items with masked images. Returns (items, status string)."""
    path = _ann_path(cfg)
    if not path.exists():
        return [], "not_available: no annotation file (signal B skipped)"
    ann = json.loads(path.read_text())
    by_id = {r["sample_id"]: r for r in diag_rows}
    sb = cfg["signal_b"]
    items = []
    for it in ann["items"][: sb["max_items"]]:
        if not (it.get("reviewed") and it.get("relevant_boxes") and it.get("control_boxes")):
            continue
        r = by_id.get(it["sample_id"])
        if r is None:
            continue
        img = load_image(r)
        rel = apply_mask(img, it["relevant_boxes"], sb["fill_color"])
        ctl = apply_mask(img, it["control_boxes"], sb["fill_color"])
        items.append({"sample_id": it["sample_id"], "mask_relevant": rel, "mask_relevant_hash": image_hash(rel),
                      "mask_control": ctl, "mask_control_hash": image_hash(ctl)})
    if not items:
        return [], "not_available: no reviewed annotations (signal B skipped)"
    return items, f"ok: {len(items)} reviewed items"


def _write_annotator(cfg: dict, rows: list[dict]) -> None:
    """Minimal in-browser box annotator: drag to draw, exports the annotation JSON."""
    out = output_dir(cfg) / "signal_b"
    out.mkdir(parents=True, exist_ok=True)
    data = [{"sample_id": r["sample_id"], "src": "../data/" + r["image_path"], "w": r["image_size"][0],
             "h": r["image_size"][1], "question": r["question"], "answers": r["answers"]} for r in rows]
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Signal B Annotator</title>
<style>:root{--bg:#fff;--fg:#222}@media (prefers-color-scheme: dark){:root{--bg:#181818;--fg:#ddd}}
body{background:var(--bg);color:var(--fg);font-family:system-ui;margin:16px} canvas{max-width:100%;cursor:crosshair;
border:1px solid #888} .item{margin-bottom:24px} button{margin-right:6px}</style></head><body>
<h1>Signal B evidence-box annotator</h1>
<p>Load the current annotation JSON, drag on each chart to draw boxes, then download and save it over
annotations/signal_b_boxes.json. Mode <b>relevant</b> (red) = every place the answer can be read from;
<b>control</b> (blue) = same-size area with no answer evidence (optional: <code>masks build</code> proposes one).
Coordinates are original-image pixels.</p>
<input type="file" id="load" accept=".json"> <button onclick="dl()">Download JSON</button>
<div id="items"></div>
<script>
const DATA = __DATA__;
let ann = {format:"signal_b_boxes_v1", items: DATA.map(d=>({sample_id:d.sample_id, relevant_boxes:[], control_boxes:[],
  control_source:null, reviewed:false, notes:""}))};
function byId(id){return ann.items.find(x=>x.sample_id===id)}
function render(){
  const root=document.getElementById('items'); root.innerHTML='';
  DATA.forEach(d=>{
    const it=byId(d.sample_id); if(!it) return;
    const div=document.createElement('div'); div.className='item';
    div.innerHTML=`<h3>${d.sample_id}</h3><p>${d.question} → <code>${JSON.stringify(d.answers)}</code></p>
      mode: <select><option>relevant</option><option>control</option></select>
      <button>undo last</button><button>clear</button><br>`;
    const [sel,undo,clr]=[div.querySelector('select'),...div.querySelectorAll('button')];
    const cv=document.createElement('canvas'); cv.width=d.w; cv.height=d.h; div.appendChild(cv);
    const img=new Image(); img.src=d.src; const ctx=cv.getContext('2d');
    const draw=(tmp)=>{ctx.drawImage(img,0,0); ctx.lineWidth=3;
      it.relevant_boxes.forEach(b=>{ctx.strokeStyle='red';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])});
      it.control_boxes.forEach(b=>{ctx.strokeStyle='#06c';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])});
      if(tmp){ctx.strokeStyle='orange';ctx.strokeRect(tmp[0],tmp[1],tmp[2]-tmp[0],tmp[3]-tmp[1])}};
    img.onload=()=>draw();
    let st=null; const pos=e=>{const r=cv.getBoundingClientRect();
      return [Math.round((e.clientX-r.left)*d.w/r.width),Math.round((e.clientY-r.top)*d.h/r.height)]};
    cv.onmousedown=e=>{st=pos(e)}; cv.onmousemove=e=>{if(st){const p=pos(e);draw(box(st,p))}};
    cv.onmouseup=e=>{if(!st)return; const b=box(st,pos(e)); st=null; if(b[2]-b[0]>2&&b[3]-b[1]>2){
      (sel.value==='relevant'?it.relevant_boxes:it.control_boxes).push(b);
      if(sel.value==='control') it.control_source='manual'; it.reviewed=false;} draw()};
    undo.onclick=()=>{(sel.value==='relevant'?it.relevant_boxes:it.control_boxes).pop();draw()};
    clr.onclick=()=>{it.relevant_boxes=[];it.control_boxes=[];draw()};
    root.appendChild(div);
  });
}
function box(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]}
function dl(){const blob=new Blob([JSON.stringify(ann,null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='signal_b_boxes.json'; a.click()}
document.getElementById('load').onchange=e=>{const f=e.target.files[0]; f.text().then(t=>{
  const loaded=JSON.parse(t); loaded.items.forEach(x=>{x.relevant_boxes=x.relevant_boxes||[];x.control_boxes=x.control_boxes||[]});
  ann=loaded; render()})};
render();
</script></body></html>"""
    (out / "annotate.html").write_text(page.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"wrote {out}/annotate.html")
