"""
rag_system.py
----------------
Top-level orchestrator. Running this module's `build()`:

  1. Loads & flattens the CSV into a chronological message stream
  2. Detects topic boundaries -> builds Topic Checkpoints (with summaries)
  3. Builds 100-message Checkpoints (with summaries), independent of topics
  4. Builds the chunk-level index for fine-grained retrieval
  5. Builds the RAG retriever over (topics + chunks + checkpoints)
  6. Extracts the user Persona
  7. Saves everything to artifacts/ as JSON (+ a pickle of the fitted
     retriever) so the chatbot / web app can load instantly without
     re-processing the whole dataset on every request.
"""
import json
import os
import pickle
import time

from .data_loader import load_messages
from .topic_segmentation import build_topic_segments
from .topic_checkpoints import build_topic_checkpoints
from .checkpoints_100 import build_message_checkpoints
from .retriever import build_chunks, RagRetriever
from .persona_extractor import build_persona

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def build(csv_path: str, max_rows: int = None, verbose: bool = True):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    t0 = time.time()

    if verbose:
        print(f"[1/6] Loading & flattening messages (max_rows={max_rows})...")
    messages = load_messages(csv_path, max_rows=max_rows)
    if verbose:
        print(f"      -> {len(messages)} messages loaded across "
              f"{len(set(m['day'] for m in messages))} days")

    if verbose:
        print("[2/6] Detecting topic boundaries & building topic segments...")
    segments = build_topic_segments(messages)
    if verbose:
        print(f"      -> {len(segments)} topic segments detected")

    if verbose:
        print("[3/6] Summarizing each topic segment (topic checkpoints)...")
    topic_checkpoints = build_topic_checkpoints(segments)

    if verbose:
        print("[4/6] Building 100-message checkpoints...")
    message_checkpoints = build_message_checkpoints(messages)
    if verbose:
        print(f"      -> {len(message_checkpoints)} message checkpoints")

    if verbose:
        print("[5/6] Building chunk index + RAG retriever...")
    chunks = build_chunks(messages)
    retriever = RagRetriever(topic_checkpoints, message_checkpoints, chunks)

    if verbose:
        print("[6/6] Extracting persona...")
    persona = build_persona(messages)

    # ---- persist ----
    with open(os.path.join(ARTIFACT_DIR, "topic_checkpoints.json"), "w") as f:
        json.dump(topic_checkpoints, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "message_checkpoints.json"), "w") as f:
        json.dump(message_checkpoints, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "persona.json"), "w") as f:
        json.dump(persona, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "messages.json"), "w") as f:
        json.dump(messages, f)
    with open(os.path.join(ARTIFACT_DIR, "retriever.pkl"), "wb") as f:
        pickle.dump(retriever, f)

    meta = {
        "num_messages": len(messages),
        "num_days": len(set(m["day"] for m in messages)),
        "num_topic_segments": len(segments),
        "num_message_checkpoints": len(message_checkpoints),
        "build_time_seconds": round(time.time() - t0, 2),
        "max_rows_used": max_rows,
    }
    with open(os.path.join(ARTIFACT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"Done in {meta['build_time_seconds']}s. Artifacts saved to {ARTIFACT_DIR}")

    return {
        "messages": messages,
        "segments": segments,
        "topic_checkpoints": topic_checkpoints,
        "message_checkpoints": message_checkpoints,
        "chunks": chunks,
        "retriever": retriever,
        "persona": persona,
        "meta": meta,
    }


def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "topic_checkpoints.json")) as f:
        topic_checkpoints = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "message_checkpoints.json")) as f:
        message_checkpoints = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "persona.json")) as f:
        persona = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "retriever.pkl"), "rb") as f:
        retriever = pickle.load(f)
    return {
        "topic_checkpoints": topic_checkpoints,
        "message_checkpoints": message_checkpoints,
        "persona": persona,
        "meta": meta,
        "retriever": retriever,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/conversations.csv")
    parser.add_argument("--max-rows", type=int, default=80)
    args = parser.parse_args()
    build(args.csv, max_rows=args.max_rows)
