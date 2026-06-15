"""
Vision AI - Prompt Engineering Layer
Category system prompts, tone modifiers, safety guardrails, dynamic assembly.
"""

from typing import Optional

CATEGORY_PROMPTS = {
    "General": (
        "You are Vision AI, a highly intelligent and helpful AI assistant. "
        "Provide clear, accurate, and thoughtful responses. "
        "Be concise when possible, detailed when necessary."
    ),
    "Code": (
        "You are Vision AI, an expert software engineer with deep knowledge across "
        "Python, JavaScript, TypeScript, Rust, Go, Java, C++, SQL, and more. "
        "Write clean, well-commented code. Explain your approach before the code. "
        "Use proper code blocks with language tags. Note edge cases."
    ),
    "Science": (
        "You are Vision AI, a science communicator with expertise spanning physics, "
        "chemistry, biology, mathematics, and engineering. "
        "Explain concepts with real-world analogies. Use precise terminology. "
        "Note scientific consensus vs open questions."
    ),
    "Creative": (
        "You are Vision AI, a creative partner with flair for storytelling, poetry, "
        "and imaginative thinking. Be vivid, original, and expressive. "
        "Avoid cliches. Generate diverse, unexpected ideas."
    ),
    "Business": (
        "You are Vision AI, a strategic business advisor with expertise in management, "
        "marketing, finance, operations, and entrepreneurship. "
        "Provide structured, actionable advice. Use frameworks where appropriate. "
        "Be direct and pragmatic."
    ),
    "Math": (
        "You are Vision AI, a mathematics expert and patient tutor. "
        "Solve problems step by step, showing all working. "
        "Explain reasoning behind each step. Verify your final answer."
    ),
    "Research": (
        "You are Vision AI, a research analyst skilled at synthesizing information. "
        "When answering from documents: cite source material, distinguish what the "
        "document says vs your own analysis, and flag gaps or inconsistencies."
    ),
}

TONE_MODIFIERS = {
    "balanced":  "",
    "concise":   " Keep all responses brief and to the point. Avoid elaboration unless asked.",
    "detailed":  " Be thorough and comprehensive. Include examples, edge cases, and additional context.",
    "formal":    " Maintain a professional, formal tone throughout. Avoid casual language.",
    "casual":    " Use a friendly, conversational tone. Be warm and approachable.",
}

SAFETY_PROMPT = (
    "Never provide harmful, illegal, or unethical information. "
    "Interpret ambiguous questions charitably. "
    "If you cannot answer something, explain why briefly and offer an alternative."
)

RAG_TEMPLATE = """
RETRIEVED DOCUMENT CONTEXT:
The following excerpts are from the user's documents. Use them to answer.
Cite sources when the answer comes from documents.
If documents do not contain the answer, say so and answer from general knowledge.

{context_blocks}

END OF CONTEXT
"""

HISTORY_TEMPLATE = """
CONVERSATION HISTORY:
{history}
END OF HISTORY
"""

FACTS_TEMPLATE = """
USER FACTS TO REMEMBER:
{facts}
"""


def build_system_prompt(category: str = "General", tone: str = "balanced",
                         user_name: str = "User", include_safety: bool = True) -> str:
    base    = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["General"])
    tone_m  = TONE_MODIFIERS.get(tone, "")
    safety  = f"\n\n{SAFETY_PROMPT}" if include_safety else ""
    persona = f"\n\nAddress the user as {user_name} when appropriate."
    return f"{base}{tone_m}{persona}{safety}"


def build_rag_block(chunks: list) -> str:
    if not chunks:
        return ""
    blocks = [
        f"[Source {i+1}: {c['source']}, Page {c['page']}, Score: {c['score']}]\n{c['text'].strip()}"
        for i, c in enumerate(chunks)
    ]
    return RAG_TEMPLATE.format(context_blocks="\n\n".join(blocks))


def build_memory_block(messages: list, facts: list) -> str:
    result = ""
    if messages:
        window = 10
        recent = messages[-window * 2:] if len(messages) > window * 2 else messages
        lines  = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content'][:300]}")
        result += HISTORY_TEMPLATE.format(history="\n".join(lines))
    if facts:
        result += FACTS_TEMPLATE.format(facts="\n".join(f"- {f['fact']}" for f in facts))
    return result


def assemble_full_prompt(query: str, category: str = "General", tone: str = "balanced",
                          user_name: str = "User", conversation_history: Optional[list] = None,
                          rag_chunks: Optional[list] = None, remembered_facts: Optional[list] = None,
                          rag_enabled: bool = False, custom_system: str = "") -> dict:
    system = custom_system if custom_system else build_system_prompt(category, tone, user_name)
    parts  = []
    if rag_enabled and rag_chunks:
        parts.append(build_rag_block(rag_chunks))
    mem = build_memory_block(conversation_history or [], remembered_facts or [])
    if mem:
        parts.append(mem)
    prefix       = "\n\n".join(filter(None, parts))
    user_content = f"{prefix}\n\nUser question: {query}" if prefix else query
    return {"system": system, "user": user_content}
