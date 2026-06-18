"""
app.py
---------
Flask web app that exposes:
  - GET  /                 -> chat + persona dashboard UI
  - POST /api/chat          -> {"query": "..."} -> chatbot answer (JSON)
  - GET  /api/persona       -> full persona JSON
  - GET  /api/topics        -> topic checkpoints JSON
  - GET  /api/checkpoints   -> 100-message checkpoints JSON
  - GET  /api/meta          -> dataset / build metadata

On first run (if artifacts/ is empty) it automatically builds the RAG
index + persona from data/conversations.csv. Set MAX_ROWS to control how
many CSV rows ("days") to ingest (default 80, fast demo-sized build;
set to a larger number / leave unset-with-None for the full dataset --
see README for timing notes on the full ~11,000-row dataset).
"""
import os
import json

from flask import Flask, render_template, request, jsonify

from src.rag_system import build, load_artifacts, ARTIFACT_DIR
from src.chatbot import Chatbot

app = Flask(__name__)

CSV_PATH = os.environ.get("CSV_PATH", os.path.join("data", "conversations.csv"))
MAX_ROWS_ENV = os.environ.get("MAX_ROWS", "80")
MAX_ROWS = None if MAX_ROWS_ENV.lower() in ("none", "all", "") else int(MAX_ROWS_ENV)

_bot = None


def get_bot():
    global _bot
    if _bot is None:
        if not os.path.exists(os.path.join(ARTIFACT_DIR, "persona.json")):
            print(f"No artifacts found -- building from {CSV_PATH} (max_rows={MAX_ROWS})...")
            build(CSV_PATH, max_rows=MAX_ROWS)
        _bot = Chatbot()
    return _bot


@app.route("/")
def index():
    bot = get_bot()
    return render_template(
        "index.html",
        persona=bot.persona,
        meta=bot.meta,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    bot = get_bot()
    result = bot.ask(query)
    return jsonify(result)


@app.route("/api/persona")
def persona():
    return jsonify(get_bot().persona)


@app.route("/api/topics")
def topics():
    with open(os.path.join(ARTIFACT_DIR, "topic_checkpoints.json")) as f:
        return jsonify(json.load(f))


@app.route("/api/checkpoints")
def checkpoints():
    with open(os.path.join(ARTIFACT_DIR, "message_checkpoints.json")) as f:
        return jsonify(json.load(f))


@app.route("/api/meta")
def meta():
    return jsonify(get_bot().meta)


@app.route("/api/rebuild", methods=["POST"])
def rebuild():
    """Force a rebuild of all artifacts (useful after changing MAX_ROWS / the CSV)."""
    global _bot
    data = request.get_json(silent=True) or {}
    max_rows = data.get("max_rows", MAX_ROWS)
    build(CSV_PATH, max_rows=max_rows)
    _bot = None
    return jsonify({"status": "rebuilt", "max_rows": max_rows})


if __name__ == "__main__":
    get_bot()  # build/load on startup so the first request isn't slow
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
