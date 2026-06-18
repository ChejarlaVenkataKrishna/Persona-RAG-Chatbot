"""
eval_topic_segmentation.py
------------------------------
Standalone sanity-check for the topic segmentation algorithm.

This dataset happens to be structured as one independent, self-contained
conversation per CSV row, so "a new row started" is a reasonable (if
imperfect) proxy ground truth for "a new topic started" -- it lets us
sanity-check precision/recall of detect_topic_boundaries() without any
manual labeling. The day/row index is used ONLY here, for evaluation; the
segmentation algorithm itself never sees it (see topic_segmentation.py).

Run:
    python eval_topic_segmentation.py --max-rows 80
"""
import argparse
from src.data_loader import load_messages
from src.topic_segmentation import detect_topic_boundaries


def evaluate(messages, tolerance=1):
    day_boundaries = set()
    prev_day = None
    for m in messages:
        if m["day"] != prev_day:
            day_boundaries.add(m["global_id"])
            prev_day = m["day"]

    detected = set(detect_topic_boundaries(messages))

    def matches(idx, truth_set):
        return any(abs(idx - t) <= tolerance for t in truth_set)

    tp = sum(1 for d in detected if matches(d, day_boundaries))
    precision = tp / len(detected) if detected else 0.0
    recall = sum(1 for t in day_boundaries if matches(t, detected)) / len(day_boundaries)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "num_messages": len(messages),
        "true_boundaries": len(day_boundaries),
        "detected_boundaries": len(detected),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/conversations.csv")
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--tolerance", type=int, default=1,
                         help="how many messages of slack to allow when matching a detected boundary to a true one")
    args = parser.parse_args()

    messages = load_messages(args.csv, max_rows=args.max_rows)
    metrics = evaluate(messages, tolerance=args.tolerance)
    print(f"Evaluated on {metrics['num_messages']} messages "
          f"({metrics['true_boundaries']} true conversation boundaries)")
    print(f"Detected boundaries : {metrics['detected_boundaries']}")
    print(f"Precision           : {metrics['precision']}")
    print(f"Recall              : {metrics['recall']}")
    print(f"F1                  : {metrics['f1']}")
