from langchain_community.document_loaders import UnstructuredPDFLoader


class DoclingLoader:

    def load(self, path):
        loader = UnstructuredPDFLoader(path)
        return loader.load()