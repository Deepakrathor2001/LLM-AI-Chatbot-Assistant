"""
Vision AI - LLM Chatbot 
"""

import os, sys, uuid, json, datetime, io, csv
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
import plotly.graph_objects as go
import pandas as pd

# ─── Path & imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

load_dotenv(BASE_DIR / ".env", override=True)

from rag_engine    import (ingest_pdf, retrieve_context, list_indexed_documents,
                            delete_document, get_vector_store_stats)
from memory_system import (get_short_term_context, save_session_to_long_term,
                            load_all_sessions, delete_session_from_memory,
                            rename_session, pin_session,
                            store_fact, get_facts, clear_facts,
                            get_profile, update_profile, increment_profile_stats,
                            get_memory_stats, record_usage, get_analytics_range,
                            get_prompts, save_prompt, edit_prompt,
                            delete_prompt, toggle_favorite_prompt)
from prompt_engine import (assemble_full_prompt, CATEGORY_PROMPTS, TONE_MODIFIERS)

# ─── Constants 
DEFAULT_MODEL = "gemini-2.5-flash"
UPLOADS_DIR   = BASE_DIR / "data" / "uploads"
EXPORTS_DIR   = BASE_DIR / "data" / "exports"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CATEGORIES    = list(CATEGORY_PROMPTS.keys())
TONES         = list(TONE_MODIFIERS.keys())

# ─── Page config
st.set_page_config(
    page_title="Vision AI",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state 
def ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

ss("messages",          [])
ss("session_id",        str(uuid.uuid4())[:8])
ss("session_name",      "")
ss("category",          "General")
ss("tone",              "balanced")
ss("rag_enabled",       False)
ss("rag_source_filter", None)
ss("theme",             "dark")
ss("user_name",         get_profile("default").get("name", "User"))
ss("active_page",       "Chat")
ss("search_query",      "")
ss("edit_pid",          None)
ss("show_new_prompt",   False)
ss("last_response",     "")
ss("web_search_on",     False)

# ─── Gemini API
def get_api_key():
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        st.error("Missing GOOGLE_API_KEY in .env file.")
        st.stop()
    return key

def get_client():
    return genai.Client(api_key=get_api_key())

# ─── Export helpers
def export_txt() -> str:
    lines = [f"Vision AI Export\nDate: {datetime.datetime.now():%Y-%m-%d %H:%M}\n{'='*50}\n"]
    for m in st.session_state.messages:
        role = "You" if m["role"] == "user" else "Vision AI"
        lines.append(f"[{m.get('ts','')}] {role}:\n{m['content']}\n")
    return "\n".join(lines)

def export_csv() -> str:
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["timestamp", "role", "content"])
    for m in st.session_state.messages:
        w.writerow([m.get("ts",""), m["role"], m["content"]])
    return buf.getvalue()

def export_json() -> str:
    return json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)


#  SIDEBAR

with st.sidebar:

    # Brand
    st.title("Vision AI")
    st.caption("From Google Gemini API")
    st.divider()

    # New Chat button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        if st.session_state.messages:
            save_session_to_long_term(
                st.session_state.session_id, st.session_state.messages,
                st.session_state.category, st.session_state.session_name)
            increment_profile_stats("default", sessions=1)
        st.session_state.messages     = []
        st.session_state.session_id   = str(uuid.uuid4())[:8]
        st.session_state.session_name = ""
        st.session_state.active_page  = "Chat"
        st.rerun()

    # Search
    sq = st.text_input("Search chats", value=st.session_state.search_query,
                        placeholder="Search...", label_visibility="visible")
    if sq != st.session_state.search_query:
        st.session_state.search_query = sq

    st.divider()

    # Navigation
    st.caption("WORKSPACE")
    pages = ["Chat", "Knowledge Base", "Memory", "Prompt Library", "Analytics", "Settings"]
    for p in pages:
        is_active = st.session_state.active_page == p
        btn_type  = "primary" if is_active else "secondary"
        if st.button(p, key=f"nav_{p}", use_container_width=True, type=btn_type):
            st.session_state.active_page = p
            st.rerun()

    st.divider()

    # Recents
    st.caption("RECENTS")
    all_sess = load_all_sessions()
    sq_lower = st.session_state.search_query.lower()

    pinned_list   = [(s, d) for s, d in all_sess.items() if d.get("pinned")]
    unpinned_list = [(s, d) for s, d in all_sess.items() if not d.get("pinned")]
    sorted_sess   = pinned_list + list(reversed(unpinned_list))

    if sorted_sess:
        for sid, sdata in sorted_sess[:10]:
            msgs    = sdata.get("messages", [])
            name    = sdata.get("name") or next(
                (m["content"][:40] for m in msgs if m["role"] == "user"), sid)
            pinned  = sdata.get("pinned", False)
            display = (">> " if pinned else "") + name[:34]
            if sq_lower and sq_lower not in display.lower():
                continue
            col_a, col_b = st.columns([5, 1])
            with col_a:
                if st.button(display, key=f"open_{sid}", use_container_width=True):
                    st.session_state.messages     = msgs
                    st.session_state.session_id   = sid
                    st.session_state.session_name = sdata.get("name", "")
                    st.session_state.category     = sdata.get("category", "General")
                    st.session_state.active_page  = "Chat"
                    st.rerun()
            with col_b:
                if st.button("x", key=f"del_{sid}"):
                    delete_session_from_memory(sid)
                    if st.session_state.session_id == sid:
                        st.session_state.messages   = []
                        st.session_state.session_id = str(uuid.uuid4())[:8]
                    st.rerun()
    else:
        st.caption("No recent chats")



#  MAIN PAGES
page = st.session_state.active_page


#  PAGE: CHAT
if page == "Chat":

    # Header
    hcol1, hcol2 = st.columns([3, 2])
    with hcol1:
        badges = "Vision AI"
        if st.session_state.rag_enabled:
            badges += "  [RAG]"
        if st.session_state.web_search_on:
            badges += "  [WEB]"
        st.title(badges)
        st.caption(f"Category: {st.session_state.category}  |  Tone: {st.session_state.tone}  |  [MEM]")
    with hcol2:
        new_name = st.text_input("Rename this chat", value=st.session_state.session_name,
                                  placeholder="Name this chat...")
        if new_name != st.session_state.session_name:
            st.session_state.session_name = new_name
            rename_session(st.session_state.session_id, new_name)

    # Action bar
    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
    with ac1:
        pinned = load_all_sessions().get(st.session_state.session_id, {}).get("pinned", False)
        if st.button("Pin" if not pinned else "Unpin", use_container_width=True):
            save_session_to_long_term(st.session_state.session_id,
                                      st.session_state.messages, st.session_state.category,
                                      st.session_state.session_name)
            pin_session(st.session_state.session_id, not pinned)
            st.rerun()
    with ac2:
        if st.session_state.messages:
            st.download_button("Export TXT", export_txt(),
                               file_name=f"vision_ai_{st.session_state.session_id}.txt",
                               mime="text/plain", use_container_width=True)
    with ac3:
        if st.session_state.messages:
            st.download_button("Export CSV", export_csv(),
                               file_name=f"vision_ai_{st.session_state.session_id}.csv",
                               mime="text/csv", use_container_width=True)
    with ac4:
        if st.session_state.messages:
            st.download_button("Export JSON", export_json(),
                               file_name=f"vision_ai_{st.session_state.session_id}.json",
                               mime="application/json", use_container_width=True)
    with ac5:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Welcome screen
    if not st.session_state.messages:
        st.subheader("What can I help with today?")
        st.write("Vision AI is powered by Google Gemini with document RAG, "
                 "persistent memory, prompt library, analytics, and more.")
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**Document RAG**")
                st.caption("Upload PDFs and get grounded, cited answers from your documents.")
            with st.container(border=True):
                st.markdown("**Export Chat**")
                st.caption("Download conversations as TXT, CSV, or JSON.")
        with c2:
            with st.container(border=True):
                st.markdown("**Memory System**")
                st.caption("Chats persist. Store facts for long-term personal context.")
            with st.container(border=True):
                st.markdown("**Dark / Light**")
                st.caption("Switch themes from Settings. Your preference is saved.")
        with c3:
            with st.container(border=True):
                st.markdown("**Prompt Library**")
                st.caption("Resume review, coding assistant, email writer and more.")
            with st.container(border=True):
                st.markdown("**Analytics**")
                st.caption("Track daily, weekly, monthly usage with charts.")
    else:
        # Render messages using st.chat_message
        for msg in st.session_state.messages:
            role    = msg["role"]
            content = msg["content"]
            ts      = msg.get("ts", "")
            sources = msg.get("sources", [])

            with st.chat_message(role):
                st.write(content)
                if ts:
                    st.caption(ts)
                if sources:
                    source_text = "  |  ".join(
                        f"{s['source']} p.{s['page']} (score: {s['score']})"
                        for s in sources
                    )
                    st.caption(f"Sources: {source_text}")

    # Copy last response
    if st.session_state.last_response:
        with st.expander("Copy Last Response"):
            st.code(st.session_state.last_response, language=None)

    # Chat input
    query = st.chat_input("Message Vision AI")
    if query:
        ts      = datetime.datetime.now().strftime("%H:%M")
        api_key = get_api_key()
        client  = get_client()
        model   = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        st.session_state.messages.append({"role": "user", "content": query, "ts": ts})

        rag_chunks = []
        if st.session_state.rag_enabled:
            with st.spinner("Searching documents..."):
                rag_chunks = retrieve_context(query, api_key, top_k=5,
                                               source_filter=st.session_state.rag_source_filter)

        prompt_pkg = assemble_full_prompt(0
            query=query, category=st.session_state.category,
            tone=st.session_state.tone, user_name=st.session_state.user_name,
            conversation_history=st.session_state.messages[-12:],
            rag_chunks=rag_chunks, remembered_facts=get_facts(5),
            rag_enabled=st.session_state.rag_enabled,
        )
        full_prompt = f"{prompt_pkg['system']}\n\n{prompt_pkg['user']}"

        with st.spinner("Thinking..."):
            try:
                resp   = client.models.generate_content(model=model, contents=full_prompt)
                answer = resp.text or "No response received."
                if not isinstance(answer, str):
                    answer = str(answer)
            except Exception as exc:
                answer = f"Error: {exc}"

        ai_ts = datetime.datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "ts": ai_ts,
            "sources": rag_chunks[:3] if rag_chunks else [],
        })
        st.session_state.last_response = answer
        tokens_approx = len(query.split()) + len(answer.split())
        increment_profile_stats("default", messages=2)
        record_usage(st.session_state.category, tokens_approx)
        save_session_to_long_term(st.session_state.session_id, st.session_state.messages,
                                   st.session_state.category, st.session_state.session_name)
        st.rerun()



#  PAGE: KNOWLEDGE BASE

elif page == "Knowledge Base":
    st.title("Knowledge Base")
    st.caption("Upload documents and enable RAG to ground answers in your files.")
    st.divider()

    st.session_state.rag_enabled = st.toggle(
        "Enable RAG for Chat", value=st.session_state.rag_enabled)

    st.subheader("Upload Document")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded:
        save_path = UPLOADS_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner(f"Indexing {uploaded.name}..."):
            result = ingest_pdf(save_path, get_api_key())
        if result["status"] == "success":
            st.success(f"Indexed {result['chunks']} chunks from {result['pages']} pages.")
        else:
            st.error(f"Error: {result['message']}")

    st.divider()
    st.subheader("Indexed Documents")
    docs = list_indexed_documents(get_api_key())
    vs   = get_vector_store_stats()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Documents", vs["total_documents"])
    with c2:
        st.metric("Total Chunks", vs["total_chunks"])

    if docs:
        src_opts = ["All Documents"] + [d["name"] for d in docs]
        src_pick = st.selectbox("Filter source for RAG", src_opts)
        st.session_state.rag_source_filter = None if src_pick == "All Documents" else src_pick
        st.divider()
        for doc in docs:
            dc1, dc2 = st.columns([5, 1])
            with dc1:
                with st.container(border=True):
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"ID: {doc['file_id']}")
            with dc2:
                st.write("")
                if st.button("Delete", key=f"kdel_{doc['file_id']}", use_container_width=True):
                    delete_document(doc["file_id"])
                    st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF above.")

    st.divider()
    st.subheader("Document Q&A")
    st.caption("Search directly in your indexed documents without going through chat.")
    doc_q = st.text_input("Ask your documents", placeholder="e.g. What is the conclusion?")
    if st.button("Search Documents") and doc_q:
        if not docs:
            st.warning("No documents indexed yet.")
        else:
            chunks = retrieve_context(doc_q, get_api_key(), top_k=5)
            if chunks:
                for i, c in enumerate(chunks, 1):
                    with st.container(border=True):
                        st.markdown(f"**Result {i} — {c['source']}  |  Page {c['page']}  |  Score: {c['score']}**")
                        st.write(c["text"])
            else:
                st.info("No relevant content found in indexed documents.")


#  PAGE: MEMORY

elif page == "Memory":
    st.title("Memory System")
    st.caption("Manage conversation history, long-term facts, and session data.")
    st.divider()

    mem_stats = get_memory_stats()
    profile   = get_profile("default")

    # Stat metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Sessions", mem_stats["total_sessions"])
    with mc2:
        st.metric("Stored Facts", mem_stats["total_facts"])
    with mc3:
        st.metric("Total Messages", profile.get("total_messages", 0))
    with mc4:
        st.metric("Total Sessions", profile.get("total_sessions", 0))

    st.divider()

    # Long-term facts
    st.subheader("Long-Term Facts")
    st.caption("Facts stored here are injected into every conversation as context.")
    facts = get_facts(50)
    if facts:
        for f in facts:
            with st.container(border=True):
                st.markdown(f"**{f['fact']}**")
                st.caption(f"{f['timestamp'][:10]}  |  source: {f['source']}")
    else:
        st.info("No facts stored yet.")

    fi1, fi2 = st.columns([4, 1])
    with fi1:
        new_fact = st.text_input("New fact", placeholder="e.g. I prefer Python over JavaScript",
                                  label_visibility="collapsed")
    with fi2:
        if st.button("Store", use_container_width=True) and new_fact:
            store_fact(new_fact)
            st.rerun()
    if facts and st.button("Clear All Facts"):
        clear_facts()
        st.rerun()

    st.divider()

    # Session history
    st.subheader("Session History")
    all_sess = load_all_sessions()
    if all_sess:
        for sid, sdata in list(reversed(list(all_sess.items())))[:20]:
            msgs   = sdata.get("messages", [])
            name   = sdata.get("name") or next(
                (m["content"][:50] for m in msgs if m["role"] == "user"), sid)
            cat    = sdata.get("category", "")
            saved  = sdata.get("saved_at", "")[:10]
            pinned = sdata.get("pinned", False)
            sc1, sc2, sc3, sc4 = st.columns([4, 1, 1, 1])
            with sc1:
                with st.container(border=True):
                    label = (">> [Pinned]  " if pinned else "") + name[:60]
                    st.markdown(f"**{label}**")
                    st.caption(f"{len(msgs)//2} exchanges  |  {cat}  |  {saved}")
            with sc2:
                if st.button("Load", key=f"mload_{sid}", use_container_width=True):
                    st.session_state.messages    = msgs
                    st.session_state.session_id  = sid
                    st.session_state.category    = cat
                    st.session_state.active_page = "Chat"
                    st.rerun()
            with sc3:
                if st.button("Pin" if not pinned else "Unpin",
                              key=f"mpin_{sid}", use_container_width=True):
                    pin_session(sid, not pinned)
                    st.rerun()
            with sc4:
                if st.button("Del", key=f"mdel_{sid}", use_container_width=True):
                    delete_session_from_memory(sid)
                    st.rerun()
    else:
        st.info("No saved sessions yet.")



#  PAGE: PROMPT LIBRARY

elif page == "Prompt Library":
    st.title("Prompt Library")
    st.caption("Manage, search, and use pre-built prompts for common tasks.")
    st.divider()

    prompts = get_prompts()

    # Filter row
    pcol1, pcol2, pcol3 = st.columns([2, 2, 1])
    with pcol1:
        filter_cat = st.selectbox("Category", ["All"] + CATEGORIES)
    with pcol2:
        filter_name = st.text_input("Search", placeholder="Search prompts...")
    with pcol3:
        show_fav = st.toggle("Favorites only")

    # Create new prompt
    if st.button("+ Create New Prompt"):
        st.session_state.show_new_prompt = not st.session_state.show_new_prompt
        st.session_state.edit_pid = None

    if st.session_state.show_new_prompt:
        with st.container(border=True):
            st.subheader("New Prompt")
            nt = st.text_input("Title", placeholder="Prompt title", key="new_title")
            nc = st.selectbox("Category", CATEGORIES, key="new_pcat")
            nx = st.text_area("Prompt text", placeholder="Enter the prompt...",
                               height=100, key="new_text")
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("Save Prompt", use_container_width=True, type="primary") and nt and nx:
                    save_prompt(nt, nx, nc)
                    st.session_state.show_new_prompt = False
                    st.rerun()
            with sc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_new_prompt = False
                    st.rerun()

    # Edit form
    if st.session_state.edit_pid:
        ep = next((p for p in prompts if p["id"] == st.session_state.edit_pid), None)
        if ep:
            with st.container(border=True):
                st.subheader("Edit Prompt")
                et = st.text_input("Title", value=ep["title"], key="ep_title")
                ec = st.selectbox("Category", CATEGORIES,
                                   index=CATEGORIES.index(ep.get("category", "General")),
                                   key="ep_cat")
                ex = st.text_area("Prompt text", value=ep["text"],
                                   height=100, key="ep_text")
                ec1, ec2 = st.columns(2)
                with ec1:
                    if st.button("Update", use_container_width=True, type="primary") and et and ex:
                        edit_prompt(st.session_state.edit_pid, et, ex, ec)
                        st.session_state.edit_pid = None
                        st.rerun()
                with ec2:
                    if st.button("Cancel Edit", use_container_width=True):
                        st.session_state.edit_pid = None
                        st.rerun()

    st.divider()

    # Filter prompts
    filtered = prompts
    if filter_cat != "All":
        filtered = [p for p in filtered if p.get("category") == filter_cat]
    if filter_name:
        filtered = [p for p in filtered if filter_name.lower() in p["title"].lower()
                    or filter_name.lower() in p["text"].lower()]
    if show_fav:
        filtered = [p for p in filtered if p.get("favorite")]

    if not filtered:
        st.info("No prompts found. Create one above.")
    else:
        for p in filtered:
            fav_star = "★" if p.get("favorite") else "☆"
            pc1, pc2, pc3, pc4, pc5 = st.columns([4, 1, 1, 1, 1])
            with pc1:
                with st.container(border=True):
                    st.markdown(f"**{fav_star} {p['title']}**")
                    st.caption(f"Category: {p.get('category','General')}")
                    st.write(p["text"][:130] + "...")
            with pc2:
                st.write("")
                if st.button("Use", key=f"puse_{p['id']}", use_container_width=True,
                              type="primary"):
                    st.session_state.active_page = "Chat"
                    st.session_state.messages.append({
                        "role": "user", "content": p["text"],
                        "ts": datetime.datetime.now().strftime("%H:%M")
                    })
                    st.rerun()
            with pc3:
                st.write("")
                if st.button(fav_star, key=f"pfav_{p['id']}", use_container_width=True):
                    toggle_favorite_prompt(p["id"])
                    st.rerun()
            with pc4:
                st.write("")
                if st.button("Edit", key=f"pedit_{p['id']}", use_container_width=True):
                    st.session_state.edit_pid        = p["id"]
                    st.session_state.show_new_prompt = False
                    st.rerun()
            with pc5:
                st.write("")
                if st.button("Del", key=f"pdel_{p['id']}", use_container_width=True):
                    delete_prompt(p["id"])
                    st.rerun()



#  PAGE: ANALYTIC
elif page == "Analytics":
    st.title("Analytics Dashboard")
    st.caption("Track your usage across daily, weekly, and monthly periods.")
    st.divider()

    period   = st.radio("Period", ["Daily (7d)", "Weekly (30d)", "Monthly (90d)"],
                         horizontal=True)
    days_map = {"Daily (7d)": 7, "Weekly (30d)": 30, "Monthly (90d)": 90}
    days     = days_map[period]
    data     = get_analytics_range(days)
    df       = pd.DataFrame(data)

    if df.empty or df["messages"].sum() == 0:
        st.info("No usage data yet. Start chatting to see your analytics.")
    else:
        total_msgs   = int(df["messages"].sum())
        total_tokens = int(df["tokens"].sum())
        active_days  = int((df["messages"] > 0).sum())
        avg_per_day  = round(total_msgs / max(active_days, 1), 1)

        # Stat metrics using st.metric
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Messages", total_msgs)
        with mc2:
            st.metric("Approx Tokens", f"{total_tokens:,}")
        with mc3:
            st.metric("Active Days", active_days)
        with mc4:
            st.metric("Avg / Day", avg_per_day)

        st.divider()

        # Line chart — messages over time
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df["date"], y=df["messages"],
            mode="lines+markers", name="Messages",
            line=dict(color="#6c8ef5", width=2.5),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(108,142,245,0.1)",
        ))
        fig_line.update_layout(
            title="Messages Over Time",
            xaxis_title="Date", yaxis_title="Messages",
            height=260, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Bar chart — token usage
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df["date"], y=df["tokens"],
            name="Tokens", marker_color="#a78bfa", opacity=0.85,
        ))
        fig_bar.update_layout(
            title="Token Usage Over Time",
            xaxis_title="Date", yaxis_title="Tokens",
            height=240, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart + day-of-week bar chart side by side
        all_cats = {}
        for rec in data:
            for cat, cnt in rec.get("categories", {}).items():
                all_cats[cat] = all_cats.get(cat, 0) + cnt

        if all_cats:
            col_pie, col_heat = st.columns(2)

            with col_pie:
                fig_pie = go.Figure(go.Pie(
                    labels=list(all_cats.keys()),
                    values=list(all_cats.values()),
                    hole=0.45,
                    marker=dict(colors=["#6c8ef5","#a78bfa","#34d399",
                                         "#fbbf24","#f87171","#f0abfc","#67e8f9"]),
                ))
                fig_pie.update_layout(
                    title="Category Distribution",
                    height=260, margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_heat:
                days_order = ["Monday","Tuesday","Wednesday",
                               "Thursday","Friday","Saturday","Sunday"]
                df["dow"]  = pd.to_datetime(df["date"]).dt.day_name()
                heat_df    = df.groupby("dow")["messages"].sum().reindex(
                    days_order, fill_value=0)
                fig_heat = go.Figure(go.Bar(
                    x=heat_df.index, y=heat_df.values,
                    marker_color="#34d399", opacity=0.85,
                ))
                fig_heat.update_layout(
                    title="Activity by Day of Week",
                    xaxis_title="", yaxis_title="Messages",
                    height=260, margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_heat, use_container_width=True)



#  PAGE: SETTINGS

elif page == "Settings":
    st.title("Settings")
    st.caption("Manage your profile, preferences, and API configuration.")
    st.divider()

    # Appearance
    st.subheader("Appearance")
    theme_pick = st.radio("Theme", ["dark", "light"],
                           index=["dark","light"].index(st.session_state.theme),
                           horizontal=True)
    if theme_pick != st.session_state.theme:
        st.session_state.theme = theme_pick
        st.rerun()

    st.divider()

    # Profile
    st.subheader("Your Profile")
    pc1, pc2 = st.columns(2)
    with pc1:
        name_in = st.text_input("Your name", value=st.session_state.user_name)
        if name_in != st.session_state.user_name:
            st.session_state.user_name = name_in
            update_profile("default", name=name_in)
    with pc2:
        pref_cat = st.selectbox("Default category", CATEGORIES,
                                 index=CATEGORIES.index(st.session_state.category))
        if pref_cat != st.session_state.category:
            st.session_state.category = pref_cat
            update_profile("default", preferred_category=pref_cat)

    st.divider()

    # Response settings
    st.subheader("Response Settings")
    tone_pick = st.selectbox("Default tone", TONES,
                               index=TONES.index(st.session_state.tone))
    if tone_pick != st.session_state.tone:
        st.session_state.tone = tone_pick
        update_profile("default", preferred_tone=tone_pick)

    st.divider()

    # Knowledge base
    st.subheader("Knowledge Base")
    rag_toggle = st.toggle("Enable RAG by default", value=st.session_state.rag_enabled)
    if rag_toggle != st.session_state.rag_enabled:
        st.session_state.rag_enabled = rag_toggle
    vs = get_vector_store_stats()
    st.caption(f"Vector store: {vs['total_documents']} documents, {vs['total_chunks']} chunks")

    st.divider()

    # API configuration
    st.subheader("API Configuration")
    api_key_raw = os.getenv("GOOGLE_API_KEY", "")
    masked = api_key_raw[:8] + "..." + api_key_raw[-4:] if len(api_key_raw) > 12 else "Not set"
    st.code(f"GOOGLE_API_KEY = {masked}", language=None)
    st.caption(f"Loaded from: {BASE_DIR / '.env'}")

    st.divider()

    # Usage stats
    st.subheader("Usage Stats")
    profile   = get_profile("default")
    mem_stats = get_memory_stats()
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Messages", profile.get("total_messages", 0))
    with s2:
        st.metric("Sessions", profile.get("total_sessions", 0))
    with s3:
        st.metric("Facts", mem_stats["total_facts"])
    with s4:
        st.metric("Prompts", mem_stats["total_prompts"])

    st.caption(
        f"Account created: {profile.get('created_at','')[:10]}  |  "
        f"Last active: {profile.get('last_active','')[:10]}"
    )
