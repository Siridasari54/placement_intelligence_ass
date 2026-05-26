from langchain.text_splitter import RecursiveCharacterTextSplitter


class SemanticChunker:

    def chunk(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        return splitter.split_documents(docs)