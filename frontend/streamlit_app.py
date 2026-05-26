import streamlit as st
import tempfile

from app.main import AppInitializer
from app.agents.workflow import Workflow


st.title("Advanced RAG System")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    paths = []

    for file in uploaded_files:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp.write(file.read())
        paths.append(temp.name)

    vectorstore, docs = AppInitializer().initialize(paths)

    query = st.text_input("Ask Question")

    if query:

        result = Workflow().run(
            query,
            vectorstore,
            docs
        )
        st.subheader("Answer")
        st.write(result["response"])

        st.subheader("Citations")

        for citation in result["citations"]:
            st.write(citation)