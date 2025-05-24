## Features

- **Live API scraping** of current calls (only this year and later), with automatic PDF download  
- **Budget extraction** from detailed XML endpoints  
- **OCR fallback** for scanned PDF pages (via PyMuPDF + Tesseract)  
- **Deadline parsing** in English & Greek (numeric dates and spelled-out months)  
- **Merge logic**: always keeps the freshest deadline per programme code  
- **Telegram bot** supports:
  - “Show me all deadlines” or “Which calls expire in June?”  
  - Semantic‐search fallback over PDF text for arbitrary queries 

  ## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ArtemBoichuk/tg_bot.git
   cd tg_bot

2. **Create & activate a virtual environment**
    python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
Install dependencies

pip install -r requirements.txt
3. **Configure your Telegram token**  
# .env file in project root
BOT_TOKEN=123456789:blahblahblah

### Usage

Ingest data (scrape API + PDFs, extract deadlines & budgets, build FAISS index)
python ingest.py

Writes JSON snapshots to ./data/

Builds index.faiss + meta.pkl for semantic search

Start the Telegram bot
python bot.py

Polls Telegram for messages

Handles /start, deadline queries, budget queries, and free-text semantic search