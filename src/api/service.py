"""
RAG Pipeline Service - manages pipelines for different chunking strategies.
"""

from pathlib import Path
from langchain_core.documents import Document

from src.config import settings
from src.loaders import load_python_files
from src.chunking import get_chunker
from src.retrieval import RetrievalPipeline
from src.agents import create_rag_agent

from .models import ChunkingStrategy


class RAGService:
    """Service class that manages RAG pipelines for different strategies."""
    
    def __init__(self):
        self.pipelines: dict[str, RetrievalPipeline] = {}
        self.indexed_files: dict[str, list[str]] = {}
    
    def get_or_create_pipeline(self, strategy: ChunkingStrategy) -> RetrievalPipeline:
        """Get existing pipeline or create a new one."""
        key = strategy.value
        
        if key not in self.pipelines:
            chunker = get_chunker(
                strategy.value,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            self.pipelines[key] = RetrievalPipeline(chunker=chunker)
            self.indexed_files[key] = []
            
        return self.pipelines[key]
    
    def index_files(
        self,
        file_paths: list[str],
        strategy: ChunkingStrategy,
        base_path: str = "data"
    ) -> tuple[int, int]:
        """Index files using the specified strategy."""
        pipeline = self.get_or_create_pipeline(strategy)
        key = strategy.value
        
        full_paths = [str(Path(base_path) / fp) for fp in file_paths]
        docs = load_python_files(full_paths)
        
        if not docs:
            return 0, 0
        
        doc_ids = pipeline.index_documents(docs)
        self.indexed_files[key].extend(file_paths)
        
        return len(docs), len(doc_ids)
    
    def query_with_agent(
        self,
        query: str,
        strategy: ChunkingStrategy,
        k: int = 3
    ) -> tuple[str, list[Document]]:
        """Query using the RAG agent."""
        pipeline = self.get_or_create_pipeline(strategy)
        
        # Temporarily set k
        original_k = settings.RETRIEVAL_K
        settings.RETRIEVAL_K = k
        
        # initiating and running agents
        retrieval_tool = pipeline.create_retrieval_tool()
        agent = create_rag_agent(tools=[retrieval_tool])
        
        messages = [{"role": "user", "content": query}]
        result = agent.invoke({"messages": messages})
        
        # Restore original k because every time we change it, it is changed globally
        settings.RETRIEVAL_K = original_k
        
        # Extract answer and get retrieved docs
        answer = result["messages"][-1].content
        retrieved_docs = pipeline.search(query, k=k)
        
        return answer, retrieved_docs


rag_service = RAGService()