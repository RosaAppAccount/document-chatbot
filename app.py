import streamlit as st
from pathlib import Path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

# ---------- pagina-titel ----------
st.set_page_config(page_title="Document-chatbot")
st.title("Chat met je eigen documenten")

# ---------- model laden ----------
llm = HuggingFaceLLM(
    model_name="microsoft/Phi-3-mini-128k-instruct",  # licht + opensource
    dtype="float16",
    context_window=8192
)
ctx = ServiceContext.from_defaults(llm=llm)

# ---------- upload ----------
files = st.file_uploader("Sleep je PDF’s / DOCX hier", accept_multiple_files=True)

if files:
    for f in files:                # opslaan naar schijf
        Path(f.name).write_bytes(f.getbuffer())

    # index opbouwen of hergebruiken
    if Path("index").exists():
        index = VectorStoreIndex.load_from_disk("index", service_context=ctx)
    else:
        docs = SimpleDirectoryReader(".").load_data()
        index = VectorStoreIndex.from_documents(docs, service_context=ctx)
        index.save_to_disk("index")

    chat = index.as_chat_engine(system_prompt="Beantwoord in het Nederlands.")

    st.success("Index klaar – stel je vraag!")

    # ---------- chat-UI ----------
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.chat_input("Je vraag…")
    if question:
        answer = chat.chat(question).response
        st.session_state.history += [("user", question), ("assistant", answer)]

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)
