from app.pipeline.rewrite.query_rewriter import QueryRewriter
from app.pipeline.rewrite.hyde_rewriter import HyDERewriter

class RewritePipeline:
    def __init__(self, llm=None):
        self.rewriter = QueryRewriter()
        self.hyde_rewriter = HyDERewriter(llm)

    def run(self, query: str, use_hyde: bool = False) -> str:
        # First clean/normalize query
        clean_query = self.rewriter.rewrite(query)
        
        if use_hyde:
            # Generate HyDE expansion doc and append to clean query
            hyde_doc = self.hyde_rewriter.generate_hyde_document(clean_query)
            return f"{clean_query} {hyde_doc}"
            
        return clean_query
