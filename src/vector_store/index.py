from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.node_parser import SentenceSplitter
import os

class LlamaVectorStore:
    def __init__(self, cohere_api_key):
        embed_model = CohereEmbedding(api_key=cohere_api_key)
        self.context = ServiceContext.from_defaults(embed_model=embed_model)
        self.index = None
        self.nodes = []

    def build_index_from_docs(self, docs):
        splitter = SentenceSplitter(chunk_size=512)
        self.nodes = splitter.get_nodes_from_documents(docs)
        self.index = VectorStoreIndex(self.nodes, service_context=self.context)

    def query(self, query_text, top_k=3):
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query_text)
