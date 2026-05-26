import streamlit as st


class CitationUI:

    def render(self, citations):

        st.subheader("Citations")

        for citation in citations:
            st.write(citation)