#!/usr/bin/env python
# ingest.py — pipeline which scrapes RIF calls website as well as manually downloaded PDFs
# =================================================================================================================

from __future__ import annotations
import pathlib, json, re, datetime as dt, requests, pickle, faiss, fitz, tqdm
import dateparser, numpy as np, xml.etree.ElementTree as ET
from utils import embed_text

# ─────────────────────────────────────────── CONSTANTS
ROOT        = pathlib.Path(__file__).parent
PDF_DIR     = ROOT / "pdfs";   PDF_DIR.mkdir(exist_ok=True)
DATA_DIR    = ROOT / "data";   DATA_DIR.mkdir(exist_ok=True)

FRESH_JSON  = DATA_DIR / "fresh_deadlines.json"
PDF_JSON    = DATA_DIR / "pdf_deadlines.json"
MERGED_JSON = DATA_DIR / "merged_deadlines.json"

INDEX_F     = ROOT / "index.faiss"
META_F      = ROOT / "meta.pkl"

CHUNK_SZ    = 400
DIM         = 384
WIN         = 300

TODAY       = dt.date.today()
THIS_YEAR   = TODAY.year

# ──────────────────────────────────────── HELPERS
def normalize_code(code: str) -> str:
    return code.replace("/", "_").replace("\\", "_")

def iso_to_date(s: str | None) -> dt.date | None:
    try:
        return dt.datetime.fromisoformat(s).date()
    except Exception:
        return None

def fetch_budget(call_id: int) -> float | None:
    """
    If stub JSON didn’t have a budget, fetch the XML detail and pick the first <Budget> tag.
    """
    url = f"https://iris.research.org.cy/api/call/{call_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        # find any <Budget> element anywhere underneath
        node = root.find(".//Budget")
        if node is not None and node.text:
            txt = node.text.strip().replace(" ", "").replace(",", "")
            return float(txt)
    except Exception:
        pass
    return None

# ─────────────────────────────────────── 1. SCRAPE STUB API + BUDGET
API_URL = "https://iris.research.org.cy/api/call/stub?owned=false"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (RIFBot)",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://iris.research.org.cy/#!/calls",
}

def scrape_api() -> list[dict]:
    try:
        r = requests.get(API_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("🔴 API error:", e)
        FRESH_JSON.write_text("[]")
        return []

    if "application/json" not in r.headers.get("Content-Type", ""):
        print("🔴 Unexpected content-type:", r.headers.get("Content-Type"))
        FRESH_JSON.write_text("[]")
        return []

    rows: list[dict] = []
    for it in r.json():
        raw = (it.get("Code") or it.get("callCode") or "").strip()
        if not raw:
            continue
        code = normalize_code(raw)

        # main deadline (ISO string or fallback EndDate)
        ddl = iso_to_date(it.get("deadline_date") or it.get("EndDate"))
        if not ddl or ddl.year < THIS_YEAR:
            continue

        # 1) Try stub JSON’s Budget field
        budget = None
        if "Budget" in it and it["Budget"] not in (None, "", 0):
            try:
                budget = float(str(it["Budget"]).replace(" ", "").replace(",", ""))
            except:
                budget = None

        # 2) If stub JSON had no Budget, fetch the detail XML
        if budget is None:
            cid = it.get("Id")
            if isinstance(cid, int):
                budget = fetch_budget(cid)

        rows.append({
            "code":     code,
            "title":    (it.get("call_title") or it.get("Title") or "").strip(),
            "deadline": ddl.strftime("%d %b %Y"),
            "status":   "OPEN" if ddl >= TODAY else "CLOSED",
            "budget":   budget,
        })

    FRESH_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    print(f"✅ scraped {len(rows)} calls (≥{THIS_YEAR}) → {FRESH_JSON}")
    return rows

fresh_rows = scrape_api()

# ──────────────────────────────────────── 2. EXTRACT FROM PDFs (unchanged)
KW_RE    = re.compile(r"(deadline|προθεσμ|submission|closing date|λήξη)", re.I)
DATE_RE  = re.compile(
    r"(?:\d{1,2}[./-]\d{1,2}[./-]20\d{2}|\d{1,2}\s+("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\s+20\d{2})",
    re.I
)

def ocr_page(pg: fitz.Page) -> str:
   # pix = pg.get_pixmap(dpi=200)
    global _ocr_count
    _ocr_count += 1
    pix = pg.get_pixmap(dpi=200)
    import PIL.Image, pytesseract
    img = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang="ell+eng")

pdf_rows, idx, meta = [], faiss.IndexFlatIP(DIM), []

_ocr_count = 0

for pdf in PDF_DIR.glob("*.pdf"):
    code = normalize_code(pdf.stem)
    doc  = fitz.open(pdf)
    txt  = "\n".join(
        (t if (t:=pg.get_text("text").strip()) else ocr_page(pg))
        for pg in tqdm.tqdm(doc, leave=False, desc=pdf.name)
    )

    found = {
        dateparser.parse(m.group(), languages=["en","el"], settings={"DATE_ORDER":"DMY"}).date()
        for km in KW_RE.finditer(txt)
        for m  in DATE_RE.finditer(txt[max(0,km.start()-WIN): km.end()+WIN])
    }
    for d in sorted(found)[:2]:
        pdf_rows.append({"code": code, "deadline": d.strftime("%d %b %Y")})

    for i in range(0, len(txt), CHUNK_SZ):
        chunk = txt[i:i+CHUNK_SZ].strip()
        if chunk:
            idx.add(embed_text(chunk))
            meta.append({"text": chunk, "source": pdf.name})

PDF_JSON.write_text(json.dumps(pdf_rows, ensure_ascii=False, indent=2), "utf-8")
faiss.write_index(idx, str(INDEX_F))
with open(META_F, "wb") as mf:
    pickle.dump(meta, mf)
print(f"✅ extracted {len(pdf_rows)} PDF deadlines → {PDF_JSON}")

print(f"🔍 OCR was used on {_ocr_count} pages out of {sum(1 for _ in PDF_DIR.glob('*.pdf'))} PDF files.")

# ──────────────────────────────────────── 3. MERGE (keep budget)
def to_d(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%d %b %Y").date()

latest: dict[str, dt.date] = {}
budgets: dict[str, float] = {}

# seed PDF deadlines
for r in pdf_rows:
    latest[r["code"]] = to_d(r["deadline"])

# override with API deadlines & capture budgets
for r in fresh_rows:
    c, d = r["code"], to_d(r["deadline"])
    if (c not in latest) or (d > latest[c]):
        latest[c] = d
    if r.get("budget") is not None:
        budgets[c] = r["budget"]

# build final merged list
merged = []
for c, d in latest.items():
    merged.append({
        "code":     c,
        "deadline": d.strftime("%d %b %Y"),
        "status":   "OPEN" if d >= TODAY else "CLOSED",
        "budget":   budgets.get(c),
    })

merged.sort(key=lambda x: to_d(x["deadline"]))
MERGED_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")
print(f"✅ merged → {MERGED_JSON} ({len(merged)} programmes)")
