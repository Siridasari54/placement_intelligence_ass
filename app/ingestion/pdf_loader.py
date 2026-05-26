from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    def load(self, path):
        loader = PyPDFLoader(path)
        return loader.load()