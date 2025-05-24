# api/bot.py
#!/usr/bin/env python
import os
import re
import html
import pickle
import logging
import datetime as dt
import difflib
import asyncio

import faiss
import numpy as np

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
)
from starlette.requests import Request
from starlette.responses import Response
from dotenv import load_dotenv

from utils import embed_text, load_deadlines, load_missing_calls

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Load environment & token ────────────────────────────────────────────────
load_dotenv()
BOT_TOKEN   = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN in environment")

# Build our Telegram application (no polling)
app = ApplicationBuilder().token(BOT_TOKEN).build()

# ── Load FAISS index + metadata ──────────────────────────────────────────────
try:
    index = faiss.read_index("index.faiss")
    with open("meta.pkl", "rb") as mf:
        meta = pickle.load(mf)
    READY = True
except Exception:
    log.warning("FAISS index not found – run ingest.py first.")
    READY = False
    meta = []

# ── Load deadlines & missing-calls ───────────────────────────────────────────
DEADLINES = load_deadlines()       # expects dicts with "code","title","deadline","status","budget"
MISSINGS  = load_missing_calls()    # fallback for legacy entries

# ── Constants & regexes ──────────────────────────────────────────────────────
BUDGET_Q   = re.compile(r"\bbudget\b", re.I)
DEADLINE_Q = re.compile(r"\b(deadline|expire|προθεσμ|λήξ[ei])", re.I)

MONTH_ALIAS = {
    "jan":1,"january":1, "feb":2,"february":2, "mar":3,"march":3,
    "apr":4,"april":4, "may":5, "jun":6,"june":6,
    "jul":7,"july":7, "aug":8,"august":8,
    "sep":9,"sept":9,"september":9,
    "oct":10,"october":10,
    "nov":11,"november":11,
    "dec":12,"december":12,
}

# Precompute for fuzzy matching
ALL_CODES  = [r["code"] for r in DEADLINES]
ALL_TITLES = [r.get("title","") for r in DEADLINES]

# ── Helper functions ─────────────────────────────────────────────────────────
def extract_month_alias(text: str) -> str | None:
    tl = text.lower()
    for m in MONTH_ALIAS:
        if m in tl:
            return m
    return None

def extract_program_code(text: str) -> str | None:
    norm = re.sub(r"[^\w]+"," ", text.lower()).strip()
    # 1) code substring
    for code in ALL_CODES:
        if code.lower() in norm:
            return code
    # 2) title substring
    for code, title in zip(ALL_CODES, ALL_TITLES):
        if title and title.lower() in norm:
            return code
    # 3) fuzzy match over codes+titles
    cand = ALL_CODES + ALL_TITLES
    best = difflib.get_close_matches(norm, cand, n=1, cutoff=0.2)
    if best:
        m = best[0]
        for code, title in zip(ALL_CODES, ALL_TITLES):
            if m == code or m == title:
                return code
    # 4) legacy fallback
    for key,info in MISSINGS.items():
        if key in norm:
            return info.get("code") or info.get("programme")
    return None

def fmt_deadlines(month_alias: str | None = None) -> str:
    if not DEADLINES:
        return "⚠️ No deadlines loaded. Please run ingest.py."
    mn = MONTH_ALIAS.get(month_alias) if month_alias else None
    today = dt.date.today()
    out = []
    for r in DEADLINES:
        try:
            d = dt.datetime.strptime(r["deadline"], "%d %b %Y").date()
        except:
            continue
        if mn and d.month != mn:
            continue
        out.append((d, r["code"], r["status"]))
    if not out:
        return "No matching deadlines."
    out.sort(key=lambda x: x[0])
    lines = [
        f"• <b>{html.escape(code)}</b> — {d.strftime('%d %b %Y')} <i>({status})</i>"
        for d,code,status in out
    ]
    return "📅 <b>Deadlines</b>\n" + "\n".join(lines)

# ── Bot logic ────────────────────────────────────────────────────────────────
async def handle_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    low = text.lower()

    # 1️⃣ Budget queries
    if BUDGET_Q.search(low):
        code = extract_program_code(text)
        if code:
            row = next((r for r in DEADLINES if r["code"]==code), None)
            if row and row.get("budget") is not None:
                return await update.message.reply_text(
                    f"💰 <b>Budget for {html.escape(code)}</b>: €{row['budget']:,.2f}",
                    parse_mode="HTML"
                )
        return await update.message.reply_text(
            "⚠️ Sorry, I couldn’t find budget info for that programme."
        )

    # 2️⃣ Deadline queries
    if DEADLINE_Q.search(low):
        alias = extract_month_alias(text)
        return await update.message.reply_text(
            fmt_deadlines(alias),
            parse_mode="HTML"
        )

    # 3️⃣ Missing-programme fallback
    for key,info in MISSINGS.items():
        if key in low:
            return await update.message.reply_text(
                f"ℹ️ <b>{html.escape(info['programme'])}</b>\n"
                f"Deadline: {info['deadline']}\n"
                f"Budget: {info['budget']}",
                parse_mode="HTML"
            )

    # 4️⃣ Semantic search in PDFs
    if not READY:
        return await update.message.reply_text("Index not ready – run ingest.py first.")
    vec = embed_text(text)
    D,I = index.search(vec, k=5)
    parts = []
    for score, idx in zip(D[0], I[0]):
        if idx<0 or score<0.25: continue
        chunk = meta[idx]["text"]
        src   = meta[idx]["source"]
        snip = (chunk[:350]+"…") if len(chunk)>350 else chunk
        parts.append(
            f"<blockquote>{html.escape(snip)}</blockquote>\n"
            f"<i>{html.escape(src)}</i>"
        )
    if parts:
        return await update.message.reply_text(
            "\n──────────\n".join(parts),
            parse_mode="HTML"
        )
    await update.message.reply_text("Sorry, nothing relevant found.")

# register our single handler
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_update))

# ── Set webhook on cold start ────────────────────────────────────────────────
WEBHOOK_URL = os.getenv("VERCEL_URL")
if WEBHOOK_URL:
    hook = f"https://{WEBHOOK_URL}/api/bot"
else:
    hook = os.getenv("WEBHOOK_URL")
if hook:
    asyncio.get_event_loop().run_until_complete(app.bot.set_webhook(hook))
    log.info("Webhook set to %s", hook)

# ── Vercel entry point ───────────────────────────────────────────────────────
async def handler(request: Request) -> Response:
    # Only accept POST from Telegram
    if request.method != "POST":
        return Response(status_code=405)
    data   = await request.json()
    update = Update.de_json(data, app.bot)
    await app.process_update(update)
    return Response(status_code=200)
