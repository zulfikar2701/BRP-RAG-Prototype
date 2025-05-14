import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
from pathlib import Path
from chat_utils import ask_rag

st.set_page_config(
    page_title="Prototype",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------  sidebar ---------------------------------------------------------
with st.sidebar:
    # Placeholder logo --------------------------------------------------------
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown("## *NTT DATA*")

    st.markdown("---")
    top_k = st.slider("Max chunks retrieved", 2, 8, 4)
    st.caption(
        "Built with **Streamlit + vLLM + Milvus Lite**\n"
        "NTT DATA Prototype - not for production"
    )

# ----------  main chat -------------------------------------------------------
st.title("📄 RAG PROTOTYPE")

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, msg)

# replay history --------------------------------------------------------------
for role, msg in st.session_state.chat:
    align = "assistant" if role == "assistant" else "user"
    st.chat_message(align).write(msg)

# user input ------------------------------------------------------------------
prompt = st.chat_input("Ask me anything about the document…")
if prompt:
    # show user bubble immediately
    st.chat_message("user").write(prompt)
    st.session_state.chat.append(("user", prompt))

    # generate answer (streamlit shows spinner)
    with st.chat_message("assistant"), st.spinner("Thinking…"):
        answer = ask_rag(prompt, top_k=top_k)
        st.write(answer)
    st.session_state.chat.append(("assistant", answer))
