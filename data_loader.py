"""
data_loader.py
----------------
Reads the conversations CSV (one *conversation* per row, each row containing
multiple "User 1: ..." / "User 2: ..." lines) and flattens everything into a
single chronological list of messages.

Convention used (documented in README):
- Row order in the CSV = chronological day order (row 0 = day 1, row 1 = day 2, ...).
- Within a row, the lines already appear in chronological (turn) order.
- "User 1" is treated as THE USER whose persona we are building.
- "User 2" is treated as the other party (companion / assistant) in that day's chat.

Each message becomes a dict:
{
    "global_id": int,      # absolute chronological index across the whole dataset
    "day": int,            # which CSV row / "day" this message belongs to
    "turn_in_day": int,    # index of the message within that day's conversation
    "speaker": "user" | "other",
    "text": str
}
"""
import re
import pandas as pd

SPEAKER_LINE_RE = re.compile(r"^\s*User\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
# The source dataset has a handful of unfilled template placeholders like
# "[user 1 name]" / "[User 2 Name]" left over from generation. These carry
# no real signal and would pollute persona/retrieval, so we strip them.
PLACEHOLDER_RE = re.compile(
    r"[\[\(]\s*user\s*\d?\s*(?:'s)?\s*(?:name)?\s*[\]\)]", re.IGNORECASE
)


def load_raw_rows(csv_path: str):
    """Load the CSV (no header, single column) and return a list of raw strings."""
    df = pd.read_csv(csv_path, header=None)
    return df[0].astype(str).tolist()


def parse_row_to_messages(row_text: str):
    """Split one CSV row (one day's conversation) into ordered (speaker, text) tuples."""
    messages = []
    for line in row_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_LINE_RE.match(line)
        if m:
            speaker_num, content = m.group(1), m.group(2).strip()
            speaker = "user" if speaker_num == "1" else "other"
            content = PLACEHOLDER_RE.sub("", content)
            content = re.sub(r"\s{2,}", " ", content).strip()
            if content:
                messages.append((speaker, content))
        else:
            # Continuation of the previous line (rare, but be defensive)
            if messages:
                prev_speaker, prev_content = messages[-1]
                messages[-1] = (prev_speaker, prev_content + " " + line)
    return messages


def load_messages(csv_path: str, max_rows: int = None):
    """
    Returns a flat, chronologically-ordered list of message dicts across
    all rows ("days") in the CSV.

    max_rows: optional cap on how many CSV rows ("days") to ingest. Useful
              for demos / fast iteration on very large datasets.
    """
    raw_rows = load_raw_rows(csv_path)
    if max_rows is not None:
        raw_rows = raw_rows[:max_rows]

    flat_messages = []
    global_id = 0
    for day_idx, row_text in enumerate(raw_rows):
        parsed = parse_row_to_messages(row_text)
        for turn_idx, (speaker, text) in enumerate(parsed):
            flat_messages.append({
                "global_id": global_id,
                "day": day_idx,
                "turn_in_day": turn_idx,
                "speaker": speaker,
                "text": text,
            })
            global_id += 1
    return flat_messages


if __name__ == "__main__":
    msgs = load_messages("data/conversations.csv", max_rows=5)
    for m in msgs:
        print(m)
