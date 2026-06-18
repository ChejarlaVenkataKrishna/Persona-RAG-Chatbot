"""
checkpoints_100.py
---------------------
Independent of topic segmentation: every 100 MESSAGES (chronological,
across the whole dataset, not per-day) we cut a checkpoint and summarize
that fixed-size block. This gives a uniform "table of contents" of the
conversation history regardless of how topics happen to align, which is
useful for fast-forward style retrieval ("what was being discussed around
message #2300?").
"""
from .summarizer import summarize

CHECKPOINT_SIZE = 100


def build_message_checkpoints(messages, checkpoint_size: int = CHECKPOINT_SIZE):
    checkpoints = []
    for i in range(0, len(messages), checkpoint_size):
        block = messages[i:i + checkpoint_size]
        if not block:
            continue
        text = " ".join(m["text"] for m in block)
        result = summarize(text, kind="checkpoint", max_sentences=4)
        checkpoints.append({
            "checkpoint_id": len(checkpoints) + 1,
            "start": i,
            "end": i + len(block),
            "num_messages": len(block),
            "summary": result["summary"],
            "summary_method": result["method"],
        })
    return checkpoints
