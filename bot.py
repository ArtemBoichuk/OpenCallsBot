#!/usr/bin/env python
# bot.py  — Telegram bot for RIF calls
# =================================================================================================================

import os, re, html, json, pickle, logging, datetime as dt, faiss, numpy as np
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv
from utils import embed_text, load_deadlines, load_missing_calls

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

# ── Load deadlines & missing-calls fallback ──────────────────────────────────
DEADLINES = load_deadlines()       # expects data/merged_deadlines.json
MISSINGS  = load_missing_calls()

# ── Month name → number mapping (full and 3-letter) ──────────────────────────
MONTH_ALIAS = {
    "jan": 1,    "january": 1,
    "feb": 2,    "february": 2,
    "mar": 3,    "march": 3,
    "apr": 4,    "april": 4,
    "may": 5,
    "jun": 6,    "june": 6,
    "jul": 7,    "july": 7,
    "aug": 8,    "august": 8,
    "sep": 9,    "sept": 9,    "september": 9,
    "oct": 10,   "october": 10,
    "nov": 11,   "november": 11,
    "dec": 12,   "december": 12,
}

# ── Heuristic to detect deadline-style queries ────────────────────────────────
DEADLINE_Q = re.compile(r"\b(deadline|expire|προθεσμ|λήξ[ei])", re.I)


def extract_month_alias(text: str) -> str | None:
    """
    Look for any full or 3-letter month in the user’s query.
    Returns the alias key (lower-case), or None.
    """
    tl = text.lower()
    for alias in MONTH_ALIAS:
        if alias in tl:
            return alias
    return None


def fmt_deadlines(month_alias: str | None = None) -> str:
    """
    Build an HTML list of deadlines, optionally filtering to the given month_alias.
    """
    if not DEADLINES:
        return "⚠️ No deadlines loaded. Please run ingest.py."

    # Determine the target month number, if any
    month_num = MONTH_ALIAS.get(month_alias) if month_alias else None

    today = dt.date.today()
    entries: list[tuple[dt.date, str, str]] = []

    # Gather each code’s deadline
    for row in DEADLINES:
        try:
            d = dt.datetime.strptime(row["deadline"], "%d %b %Y").date()
        except ValueError:
            continue
        # If user asked for a specific month, filter
        if month_num and d.month != month_num:
            continue
        entries.append((d, row["code"], row["status"]))

    if not entries:
        return "No matching deadlines."

    # Sort by date
    entries.sort(key=lambda x: x[0])

    # Format each line
    lines = [
        f"• <b>{html.escape(code)}</b> — {d.strftime('%d %b %Y')} <i>({status})</i>"
        for d, code, status in entries
    ]
    return "📅 <b>Deadlines</b>\n" + "\n".join(lines)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Ask me about RIF calls, e.g.\n"
        "• Show me all deadlines\n"
        "• Which calls expire in June?\n"
        "• Which calls expire in Dec?\n"
        "• PRIMA budget"
    )


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    low  = text.lower()

    # 1️⃣ Deadline queries
    if DEADLINE_Q.search(low):
        m_alias = extract_month_alias(low)
        await update.message.reply_text(
            fmt_deadlines(m_alias),
            parse_mode="HTML"
        )
        return

    # 2️⃣ Missing-programme fallback
    for key, info in MISSINGS.items():
        if key in low:
            await update.message.reply_text(
                f"ℹ️ <b>{html.escape(info['programme'])}</b>\n"
                f"Deadline: {info['deadline']}\n"
                f"Budget: {info['budget']}",
                parse_mode="HTML"
            )
            return

    # 3️⃣ Semantic search in PDFs
    if not READY:
        await update.message.reply_text("Index not ready – run ingest.py first.")
        return

    vec = embed_text(text)
    D, I = index.search(vec, k=5)
    responses = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or score < 0.25:
            continue
        chunk = meta[idx]["text"]
        src   = meta[idx]["source"]
        snippet = chunk[:350] + "…" if len(chunk) > 350 else chunk
        responses.append(
            f"<blockquote>{html.escape(snippet)}</blockquote>\n"
            f"<i>{html.escape(src)}</i>"
        )

    if responses:
        await update.message.reply_text("\n──────────\n".join(responses), parse_mode="HTML")
    else:
        await update.message.reply_text("Sorry, nothing relevant found.")


def main() -> None:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN in your environment")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()


if __name__ == "__main__":
    main()
