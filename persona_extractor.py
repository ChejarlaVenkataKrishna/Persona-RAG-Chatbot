"""
persona_extractor.py
------------------------
Builds a structured persona JSON purely from SIGNALS found in the user's
("User 1") own messages -- never guessed/hallucinated. Every fact stored
carries the evidence message(s) it was extracted from, so the persona is
auditable.

Four buckets, as required by the task:
  1. habits                -> recurring routines, sleep patterns, food habits
  2. personal_facts        -> relationships, job, location, pets, life events
  3. personality_traits    -> lexicon + heuristic scored traits
  4. communication_style   -> message-length stats, punctuation/emoji usage, tone

APPROACH
----------
This is intentionally rule-based / lexicon-based (regex + keyword counting)
rather than "ask an LLM to guess a personality" -- it is slower to extend
but every single claim traces back to a literal quote, which is exactly
what "based on actual conversation signals, not guesses" calls for.
"""
import re
from collections import Counter, defaultdict

EMOJI_RE = re.compile(
    "["                     
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+", flags=re.UNICODE
)

# ---------- HABITS ----------
HABIT_PATTERNS = {
    "sleep_schedule": [
        r"\bI (?:usually |always |often )?(?:wake up|get up) (?:at |around )?([\w:apm\. ]+)",
        r"\bI(?:'m| am) a (night owl|early riser|morning person)\b",
        r"\bI (?:stay|stayed|like to stay) up (?:late|all night)\b",
        r"\bI go to (?:bed|sleep) (?:at |around )?([\w:apm\. ]+)",
    ],
    "exercise_routine": [
        r"\bI (?:usually |always |often )?(?:go to the gym|work out|workout|run|jog|do yoga|play \w+)\b",
        r"\bI (?:like|love) to (run|exercise|do yoga|play \w+|hike|swim|bike)\b",
    ],
    "food_habits": [
        r"\bI(?:'m| am) (vegan|vegetarian|pescatarian|gluten[- ]free|on a diet)\b",
        r"\bI (?:love|like) (?:to (?:cook|bake|eat)|making) ([\w' ]+)",
        r"\bmy favorite (?:food|meal|dish) is ([\w' ]+)",
    ],
    "daily_routine": [
        r"\bevery (?:day|morning|night|weekend) I ([\w' ]+)",
        r"\bI (?:usually|always|typically) ([\w' ]+) (?:before|after|every day)\b",
    ],
}

# ---------- PERSONAL FACTS ----------
FACT_PATTERNS = {
    "job_or_study": [
        r"\bI(?:'m| am) (?:a|an) ([\w' ]+?)(?:\.|,|!|$)",
        r"\bI work as (?:a|an) ([\w' ]+?)(?:\.|,|!|$)",
        r"\bI(?:'m| am) (?:studying|majoring in) ([\w' ]+?)(?:\.|,|!|$)",
    ],
    "location": [
        r"\bI(?:'m| am) (?:from|moving to|living in) ([\w' ,]+?)(?:\.|,|!|$)",
        r"\bI live in ([\w' ,]+?)(?:\.|,|!|$)",
    ],
    "relationships": [
        r"\bmy (wife|husband|girlfriend|boyfriend|fianc[ée]+|son|daughter|mom|dad|mother|father|sister|brother|best friend)\b",
        r"\bI(?:'m| am) (married|engaged|single|dating someone)\b",
    ],
    "pets": [
        r"\bI have (?:a|an|\d+) (dog|cat|dogs|cats|bird|fish|hamster|rabbit)\b",
        r"\bmy (dog|cat|pet)\b",
    ],
    "life_events": [
        r"\bI(?:'m| am) (?:getting married|moving|graduating|expecting|pregnant)\b",
        r"\bI just (got married|graduated|moved|started a new job|had a baby)\b",
    ],
    "hobbies": [
        r"\bI (?:love|enjoy|like) ([\w' ]+?ing)\b",
    ],
}

# ---------- PERSONALITY LEXICONS (tiny, fully offline) ----------
TRAIT_LEXICON = {
    "funny": ["lol", "lmao", "haha", "hilarious", "joke", "funny", "😂", "🤣"],
    "emotional": ["miss", "sad", "cry", "crying", "sorry", "love you", "heartbroken", "upset", "😢", "❤️"],
    "enthusiastic": ["awesome", "amazing", "excited", "can't wait", "love it", "so cool", "!!"],
    "serious": ["actually", "however", "therefore", "in fact", "important", "concerned"],
    "curious": ["why", "how does", "what if", "i wonder", "curious"],
    "supportive": ["i'm sure", "you got this", "that's great", "good luck", "i understand", "sounds great"],
}


def _extract_with_patterns(messages, patterns_dict):
    found = defaultdict(list)
    for msg in messages:
        text = msg["text"]
        low = text.lower()
        for label, patterns in patterns_dict.items():
            for pat in patterns:
                for m in re.finditer(pat, low, flags=re.IGNORECASE):
                    snippet = m.group(0).strip()
                    found[label].append({
                        "value": snippet,
                        "evidence": text,
                        "global_id": msg["global_id"],
                        "day": msg["day"],
                    })
    return found


def _dedupe_top(items, top_k=8):
    """Dedupe by normalized value, keep first evidence, return most frequent first."""
    counts = Counter()
    first_seen = {}
    for it in items:
        key = re.sub(r"\s+", " ", it["value"].strip().lower())
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = it
    ranked = counts.most_common(top_k)
    return [
        {
            "value": first_seen[key]["value"],
            "count": count,
            "example_quote": first_seen[key]["evidence"],
            "first_seen_day": first_seen[key]["day"],
        }
        for key, count in ranked
    ]


def extract_habits(user_messages):
    raw = _extract_with_patterns(user_messages, HABIT_PATTERNS)
    return {label: _dedupe_top(items) for label, items in raw.items() if items}


def extract_personal_facts(user_messages):
    raw = _extract_with_patterns(user_messages, FACT_PATTERNS)
    return {label: _dedupe_top(items) for label, items in raw.items() if items}


def extract_personality_traits(user_messages):
    trait_scores = Counter()
    trait_evidence = defaultdict(list)
    total_msgs = max(len(user_messages), 1)

    for msg in user_messages:
        low = msg["text"].lower()
        for trait, keywords in TRAIT_LEXICON.items():
            for kw in keywords:
                if kw in low:
                    trait_scores[trait] += 1
                    if len(trait_evidence[trait]) < 3:
                        trait_evidence[trait].append(msg["text"])

    traits = []
    for trait, score in trait_scores.most_common():
        traits.append({
            "trait": trait,
            "signal_count": score,
            "frequency_per_100_msgs": round(score / total_msgs * 100, 2),
            "example_quotes": trait_evidence[trait],
        })
    return traits


def extract_communication_style(user_messages):
    if not user_messages:
        return {}

    lengths = [len(m["text"].split()) for m in user_messages]
    exclam = sum(m["text"].count("!") for m in user_messages)
    question = sum(m["text"].count("?") for m in user_messages)
    emoji_count = sum(len(EMOJI_RE.findall(m["text"])) for m in user_messages)
    caps_words = sum(
        1 for m in user_messages for w in m["text"].split()
        if w.isupper() and len(w) > 1
    )
    slang_terms = ["lol", "omg", "u ", "ur ", "gonna", "wanna", "kinda", "yeah", "haha"]
    slang_count = sum(
        1 for m in user_messages for term in slang_terms if term in m["text"].lower()
    )

    n = len(user_messages)
    avg_len = sum(lengths) / n

    if avg_len < 8:
        length_style = "short, punchy messages"
    elif avg_len < 16:
        length_style = "medium-length, conversational messages"
    else:
        length_style = "long, detailed messages"

    return {
        "avg_message_length_words": round(avg_len, 1),
        "message_length_style": length_style,
        "exclamation_marks_per_100_msgs": round(exclam / n * 100, 1),
        "question_marks_per_100_msgs": round(question / n * 100, 1),
        "emoji_usage_per_100_msgs": round(emoji_count / n * 100, 1),
        "slang_casual_terms_per_100_msgs": round(slang_count / n * 100, 1),
        "uses_emoji": emoji_count > 0,
        "uses_exclamations_frequently": (exclam / n) > 0.3,
        "tone": (
            "casual / enthusiastic" if (exclam / n) > 0.3 or slang_count / n > 0.2
            else "neutral / measured"
        ),
    }


def build_persona(messages):
    """
    messages: full chronological message list (both speakers).
    The persona is built ONLY from speaker == "user" messages.
    """
    user_messages = [m for m in messages if m["speaker"] == "user"]

    persona = {
        "num_user_messages_analyzed": len(user_messages),
        "num_days_covered": len(set(m["day"] for m in user_messages)) if user_messages else 0,
        "habits": extract_habits(user_messages),
        "personal_facts": extract_personal_facts(user_messages),
        "personality_traits": extract_personality_traits(user_messages),
        "communication_style": extract_communication_style(user_messages),
    }
    return persona
