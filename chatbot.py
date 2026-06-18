"""
chatbot.py
-------------
Query handling layer (Part 3 of the task).

Two query "modes", auto-detected from the question:

1. PERSONA questions ("what kind of person is this user?", "what are
   their habits?", "how do they talk?") -> answered primarily from the
   persona.json structured data (Part 2), optionally enriched with a
   couple of supporting quotes pulled via retrieval so the answer stays
   evidence-grounded rather than just reciting JSON.

2. CONTENT questions (anything about what was discussed / specific facts
   from the conversation history) -> full RAG: retrieve relevant TOPIC
   summaries + relevant message CHUNKS + relevant 100-message
   CHECKPOINTS, combine them into context, and generate the final answer
   (Grok if configured, else local extractive combiner).

Every answer also returns its `sources` (which topics/chunks/checkpoints
were used) so the caller (web UI) can show "why" the bot said what it said.
"""
import re

from .rag_system import load_artifacts
from .summarizer import generate_answer

PERSONA_KEYWORDS = [
    "what kind of person", "what kind of user", "describe this user", "describe the user",
    "personality", "what are their habits", "what are his habits", "what are her habits",
    "habits", "how do they talk", "how does this user talk", "communication style",
    "how do they communicate", "what is this user like", "tell me about this user",
    "tell me about the user", "persona", "personal facts", "what do you know about",
]


def is_persona_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in PERSONA_KEYWORDS)


def _format_persona_for_answer(persona: dict, query: str) -> str:
    q = query.lower()
    parts = []

    wants_habits = "habit" in q
    wants_style = "talk" in q or "communicat" in q or "style" in q
    wants_traits = "personality" in q or "trait" in q or "kind of person" in q
    wants_facts = "fact" in q or "relationship" in q or "job" in q or "live" in q
    wants_everything = not any([wants_habits, wants_style, wants_traits, wants_facts])

    if wants_traits or wants_everything:
        traits = persona.get("personality_traits", [])[:4]
        if traits:
            trait_str = ", ".join(f"{t['trait']} ({t['signal_count']} signals)" for t in traits)
            parts.append(f"Personality: based on language patterns across their messages, "
                         f"the dominant traits are {trait_str}.")

    if wants_habits or wants_everything:
        habits = persona.get("habits", {})
        habit_bits = []
        for label, items in habits.items():
            if items:
                habit_bits.append(f"{label.replace('_', ' ')}: " + "; ".join(i["value"] for i in items[:3]))
        if habit_bits:
            parts.append("Habits: " + " | ".join(habit_bits))

    if wants_facts or wants_everything:
        facts = persona.get("personal_facts", {})
        fact_bits = []
        for label, items in facts.items():
            if items:
                fact_bits.append(f"{label.replace('_', ' ')}: " + "; ".join(i["value"] for i in items[:3]))
        if fact_bits:
            parts.append("Personal facts: " + " | ".join(fact_bits))

    if wants_style or wants_everything:
        style = persona.get("communication_style", {})
        if style:
            parts.append(
                f"Communication style: {style.get('message_length_style', 'n/a')}, "
                f"tone is {style.get('tone', 'n/a')}, "
                f"~{style.get('exclamation_marks_per_100_msgs', 0)} exclamation marks "
                f"and ~{style.get('question_marks_per_100_msgs', 0)} question marks per 100 messages"
                + (", uses emoji" if style.get("uses_emoji") else ", rarely uses emoji") + "."
            )

    if not parts:
        parts.append("I don't have enough signal in the analyzed conversations to answer that confidently.")

    return "\n\n".join(parts)


class Chatbot:
    def __init__(self):
        artifacts = load_artifacts()
        self.persona = artifacts["persona"]
        self.retriever = artifacts["retriever"]
        self.meta = artifacts["meta"]

    def ask(self, query: str) -> dict:
        if is_persona_question(query):
            answer = _format_persona_for_answer(self.persona, query)
            # also pull a couple of supporting quotes via RAG for grounding
            retrieved = self.retriever.retrieve(query, top_k_topics=1, top_k_chunks=2, top_k_checkpoints=0)
            return {
                "answer": answer,
                "mode": "persona",
                "method": "structured_persona",
                "sources": retrieved,
            }

        # content / RAG question
        retrieved = self.retriever.retrieve(query, top_k_topics=3, top_k_chunks=4, top_k_checkpoints=1)
        context_blocks = (
            [t["summary"] for t in retrieved["topics"]]
            + [c["text"] for c in retrieved["chunks"]]
            + [c["summary"] for c in retrieved["checkpoints"]]
        )
        if not context_blocks:
            return {
                "answer": "I couldn't find anything relevant to that in the conversation history.",
                "mode": "rag",
                "method": "none",
                "sources": retrieved,
            }
        result = generate_answer(query, context_blocks)
        return {
            "answer": result["answer"],
            "mode": "rag",
            "method": result["method"],
            "sources": retrieved,
        }


if __name__ == "__main__":
    bot = Chatbot()
    print(f"Loaded artifacts: {bot.meta}")
    demo_questions = [
        "What kind of person is this user?",
        "What are their habits?",
        "How do they talk?",
        "Does the user have any pets?",
        "What does the user do for work?",
    ]
    for q in demo_questions:
        print("\n" + "=" * 70)
        print("Q:", q)
        res = bot.ask(q)
        print(f"[{res['mode']} | {res['method']}]")
        print(res["answer"])
