"""
Vision AI - Memory System
Short-term buffer, long-term JSON persistence, user profiles, analytics tracking.
"""

import json
import uuid
import datetime
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "data" / "user_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE  = PROFILES_DIR / "memory_store.json"
SHORT_TERM_WINDOW = 10


# ─── Internal load/save ───────────────────────────────────────────────────────

def _load() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"facts": [], "sessions": {}, "profiles": {}, "analytics": {}, "prompts": []}


def _save(data: dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ─── Short-term context ───────────────────────────────────────────────────────

def get_short_term_context(messages: list, window: int = SHORT_TERM_WINDOW) -> str:
    recent = messages[-window * 2:] if len(messages) > window * 2 else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:300]}")
    return "\n".join(lines)


# ─── Session management ───────────────────────────────────────────────────────

def save_session_to_long_term(session_id: str, messages: list,
                               category: str = "General", name: str = ""):
    data = _load()
    preview = next((m["content"][:50] for m in messages if m["role"] == "user"), "Untitled")
    data["sessions"][session_id] = {
        "messages":     messages,
        "category":     category,
        "name":         name or preview,
        "pinned":       data["sessions"].get(session_id, {}).get("pinned", False),
        "saved_at":     datetime.datetime.now().isoformat(),
        "message_count": len(messages),
    }
    _save(data)


def load_all_sessions() -> dict:
    return _load().get("sessions", {})


def delete_session_from_memory(session_id: str):
    data = _load()
    data["sessions"].pop(session_id, None)
    _save(data)


def rename_session(session_id: str, new_name: str):
    data = _load()
    if session_id in data["sessions"]:
        data["sessions"][session_id]["name"] = new_name
        _save(data)


def pin_session(session_id: str, pinned: bool):
    data = _load()
    if session_id in data["sessions"]:
        data["sessions"][session_id]["pinned"] = pinned
        _save(data)


# ─── Facts / long-term memory ─────────────────────────────────────────────────

def store_fact(fact: str, source: str = "user"):
    data = _load()
    data["facts"].append({
        "id":        str(uuid.uuid4())[:8],
        "fact":      fact,
        "source":    source,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    _save(data)


def get_facts(limit: int = 10) -> list:
    return _load().get("facts", [])[-limit:]


def clear_facts():
    data = _load()
    data["facts"] = []
    _save(data)


# ─── User profile ─────────────────────────────────────────────────────────────

DEFAULT_PROFILE = {
    "name":               "User",
    "preferred_category": "General",
    "preferred_tone":     "balanced",
    "total_messages":     0,
    "total_sessions":     0,
    "created_at":         None,
    "last_active":        None,
}


def get_profile(user_id: str = "default") -> dict:
    data = _load()
    profiles = data.get("profiles", {})
    if user_id not in profiles:
        p = DEFAULT_PROFILE.copy()
        p["created_at"] = p["last_active"] = datetime.datetime.now().isoformat()
        profiles[user_id] = p
        data["profiles"] = profiles
        _save(data)
    return profiles[user_id]


def update_profile(user_id: str = "default", **kwargs):
    data = _load()
    profiles = data.get("profiles", {})
    if user_id not in profiles:
        profiles[user_id] = DEFAULT_PROFILE.copy()
    profiles[user_id].update(kwargs)
    profiles[user_id]["last_active"] = datetime.datetime.now().isoformat()
    data["profiles"] = profiles
    _save(data)


def increment_profile_stats(user_id: str = "default", messages: int = 0, sessions: int = 0):
    data = _load()
    profiles = data.get("profiles", {})
    if user_id not in profiles:
        profiles[user_id] = DEFAULT_PROFILE.copy()
    profiles[user_id]["total_messages"] = profiles[user_id].get("total_messages", 0) + messages
    profiles[user_id]["total_sessions"]  = profiles[user_id].get("total_sessions",  0) + sessions
    profiles[user_id]["last_active"] = datetime.datetime.now().isoformat()
    data["profiles"] = profiles
    _save(data)


# ─── Analytics ────────────────────────────────────────────────────────────────

def record_usage(category: str = "General", tokens_approx: int = 0):
    """Record a single usage event with date key."""
    data  = _load()
    today = datetime.date.today().isoformat()
    analytics = data.get("analytics", {})
    if today not in analytics:
        analytics[today] = {"messages": 0, "tokens": 0, "categories": {}}
    analytics[today]["messages"] += 1
    analytics[today]["tokens"]   += tokens_approx
    cat_map = analytics[today].get("categories", {})
    cat_map[category] = cat_map.get(category, 0) + 1
    analytics[today]["categories"] = cat_map
    data["analytics"] = analytics
    _save(data)


def get_analytics_range(days: int = 30) -> list:
    """Return list of {date, messages, tokens, categories} for last N days."""
    data      = _load()
    analytics = data.get("analytics", {})
    today     = datetime.date.today()
    result    = []
    for i in range(days - 1, -1, -1):
        d   = (today - datetime.timedelta(days=i)).isoformat()
        rec = analytics.get(d, {"messages": 0, "tokens": 0, "categories": {}})
        result.append({"date": d, **rec})
    return result


# ─── Prompt library ───────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    {"id": "p1", "title": "Resume Review",        "category": "Business",  "favorite": False,
     "text": "Please review my resume and provide detailed feedback on structure, content, and areas for improvement. Suggest specific changes to make it more impactful."},
    {"id": "p2", "title": "Interview Preparation","category": "Business",  "favorite": False,
     "text": "Help me prepare for a job interview. Ask me common interview questions one at a time, evaluate my answers, and provide constructive feedback and model answers."},
    {"id": "p3", "title": "Coding Assistant",     "category": "Code",      "favorite": True,
     "text": "You are my coding assistant. Help me write clean, well-documented code. Explain your approach, highlight edge cases, and suggest improvements."},
    {"id": "p4", "title": "Data Analysis",        "category": "Science",   "favorite": False,
     "text": "Help me analyze data. I will describe my dataset and goals. Suggest appropriate analysis methods, help interpret results, and recommend visualizations."},
    {"id": "p5", "title": "Research Assistant",   "category": "Research",  "favorite": True,
     "text": "Act as my research assistant. Help me find key information, summarize findings, identify gaps in knowledge, and structure my research logically."},
    {"id": "p6", "title": "Email Writer",         "category": "Business",  "favorite": False,
     "text": "Help me write a professional email. I will provide context about the recipient and purpose. Write a clear, concise, and appropriately toned email."},
]


def get_prompts() -> list:
    data    = _load()
    prompts = data.get("prompts", [])
    if not prompts:
        data["prompts"] = DEFAULT_PROMPTS
        _save(data)
        return DEFAULT_PROMPTS
    return prompts


def save_prompt(title: str, text: str, category: str = "General") -> str:
    data    = _load()
    prompts = data.get("prompts", [])
    pid     = str(uuid.uuid4())[:8]
    prompts.append({"id": pid, "title": title, "text": text,
                    "category": category, "favorite": False})
    data["prompts"] = prompts
    _save(data)
    return pid


def edit_prompt(pid: str, title: str, text: str, category: str):
    data    = _load()
    prompts = data.get("prompts", [])
    for p in prompts:
        if p["id"] == pid:
            p["title"]    = title
            p["text"]     = text
            p["category"] = category
            break
    data["prompts"] = prompts
    _save(data)


def delete_prompt(pid: str):
    data    = _load()
    prompts = [p for p in data.get("prompts", []) if p["id"] != pid]
    data["prompts"] = prompts
    _save(data)


def toggle_favorite_prompt(pid: str):
    data    = _load()
    prompts = data.get("prompts", [])
    for p in prompts:
        if p["id"] == pid:
            p["favorite"] = not p.get("favorite", False)
            break
    data["prompts"] = prompts
    _save(data)


# ─── Stats summary ────────────────────────────────────────────────────────────

def get_memory_stats() -> dict:
    data = _load()
    return {
        "total_sessions": len(data.get("sessions", {})),
        "total_facts":    len(data.get("facts", [])),
        "total_profiles": len(data.get("profiles", {})),
        "total_prompts":  len(data.get("prompts", [])),
    }
