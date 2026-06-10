import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Voyager", page_icon="✈️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500&display=swap');

.stApp { background-color: #0a0e1a; color: #e2e5f0; font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background-color: #0d1120; border-right: 1px solid #1e2640; }
[data-testid="stSidebar"] button {
    background-color: #131929 !important;
    color: #8892b0 !important;
    border: 1px solid #1e2640 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #1a2240 !important;
    color: #ffffff !important;
}

/* Hide menu and footer but NOT the header (which contains sidebar toggle) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Make header background transparent so it blends in */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Keep the sidebar collapse/expand toggle button always visible */
button[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #4f8ef7 !important;
}

/* Header card — compact */
/* Header card */
.voyager-header {
    background: #0d1120;
    border: 1px solid #1a2240;
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 20px;
}

.voyager-header-left h1 {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin: 0;
}

.voyager-header-left h1 span {
    color: #4f8ef7;
}

.voyager-header-left p {
    color: #7d89b0;
    font-size: 13px;
    margin: 4px 0 0 0;
}
</style>
""", unsafe_allow_html=True)

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_id" not in st.session_state:
    st.session_state.active_id = None

def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {"title": "New chat", "messages": []}
    st.session_state.active_id = cid

if st.session_state.active_id is None:
    new_chat()

# Sidebar
with st.sidebar:
    st.title("✈️ Voyager")
    if st.button("+ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("Previous conversations")

    for cid, data in reversed(list(st.session_state.conversations.items())):
        label = data["title"]
        if st.button(label, key=cid, use_container_width=True):
            st.session_state.active_id = cid
            st.rerun()

# Main chat
cid = st.session_state.active_id
convo = st.session_state.conversations[cid]

# Count stats
total_chats = len(st.session_state.conversations)
total_msgs = sum(len(c["messages"]) for c in st.session_state.conversations.values())

# Header

if len(convo["messages"]) == 0:
    st.markdown("""
    <div class="voyager-header">
        <div class="voyager-header-left">
            <h1>✈ Voy<span>ager</span></h1>
            <p>AI-powered travel planner</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in convo["messages"]:
    role = "👤" if msg["role"] == "user" else "🤖"

    st.markdown(
        f"""
        <div style="
            background:#0d1120;
            border:2px solid #1a2240;
            border-radius:12px;
            padding:12px 16px;
            margin-bottom:10px;
            color:#ffffff;
            display:flex;
            align-items:center;
            gap:12px;
        ">
            <span style="font-size:20px;">{role}</span>
            <span>{msg["content"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
prompt = st.chat_input("Where do you want to go?")

if prompt:
    convo["messages"].append({"role": "user", "content": prompt})

    if len(convo["messages"]) == 1:
        convo["title"] = prompt[:40]

    with st.spinner("Planning your trip..."):
        try:
            r = requests.post(
                API_URL,
                json={"prompt": prompt, "thread_id": cid},
                timeout=60
            )
            r.raise_for_status()
            data = r.json()

            ai = data.get("AIMessage", {})
            msgs = ai.get("message_hist", [])

            reply = ""
            for m in reversed(msgs):
                if m.get("type") == "ai":
                    reply = m.get("content", "")
                    break

            convo["messages"].append(
                {"role": "assistant", "content": reply}
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach backend. Make sure FastAPI is running on port 8000."
            )
        except Exception as e:
            st.error(str(e))
    st.rerun()