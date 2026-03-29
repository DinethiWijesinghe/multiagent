"""
Retrieval-Augmented Generation (RAG) System
===========================================
Integrates external knowledge sources with lightweight retrieval for grounded responses.

Features:
- Indexes documents, databases, and web content
- Retrieves relevant information based on user queries
- Supports keyless grounded mode (default, no model API key required)
- Optionally augments prompts with Gemini when configured
- Supports multiple data sources: JSON databases, PDFs, text files

Components:
- VectorStore: ChromaDB for document storage and retrieval
- Embeddings: Sentence Transformers for text vectorization
- Retriever: Semantic search with similarity scoring
- Generator: keyless grounded formatter (default) or optional Gemini
"""

# pyright: reportMissingImports=false

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE", "LITE").strip().upper()
RAG_LLM_PROVIDER = os.environ.get("RAG_LLM_PROVIDER", "none").strip().lower()
_PROFILE_DEFAULTS = {
    "FULL": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
    },
    "LITE": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 2,
    },
    "RESEARCH": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 2,
    },
}


def _import_symbol(module_candidates: List[str], symbol: str):
    """Import a symbol from the first available module candidate."""
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, symbol):
                return getattr(module, symbol)
        except Exception:
            continue
    raise ImportError(
        f"Could not import '{symbol}'. Tried: {', '.join(module_candidates)}"
    )


class RAGSystem:
    """RAG system for enhancing chatbot responses with external knowledge."""

    def __init__(self, persist_directory: str = "./data/vectorstore", model: str = "gemini-2.5-flash"):
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.llm_provider = RAG_LLM_PROVIDER
        self.llm = None
        self.runtime_profile = RUNTIME_PROFILE
        profile_defaults = _PROFILE_DEFAULTS.get(self.runtime_profile, _PROFILE_DEFAULTS["LITE"])
        self.default_top_k = int(os.environ.get("RAG_TOP_K", str(profile_defaults["top_k"])))
        chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", str(profile_defaults["chunk_size"])))
        chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", str(profile_defaults["chunk_overlap"])))

        # Resolve LangChain symbols across old/new package layouts.
        self.DocumentCls = _import_symbol(
            ["langchain_core.documents"],
            "Document",
        )
        self.TextSplitterCls = _import_symbol(
            ["langchain_text_splitters"],
            "RecursiveCharacterTextSplitter",
        )
        self.ChromaCls = _import_symbol(
            ["langchain_community.vectorstores"],
            "Chroma",
        )
        self.EmbeddingCls = _import_symbol(
            ["langchain_community.embeddings"],
            "SentenceTransformerEmbeddings",
        )

        # Initialize embeddings model
        self.embedding_model = self.EmbeddingCls(
            model_name="all-MiniLM-L6-v2"
        )

        # Initialize vector store
        self.vectorstore = self.ChromaCls(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_model
        )

        self._initialize_llm()

        # Initialize text splitter
        self.text_splitter = self.TextSplitterCls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        logger.info(
            "RAG system initialized with model %s, provider %s, profile %s, chunk_size=%s, chunk_overlap=%s, top_k=%s",
            self.model,
            self.llm_provider,
            self.runtime_profile,
            chunk_size,
            chunk_overlap,
            self.default_top_k,
        )

    def _initialize_llm(self) -> None:
        """Initialize an optional LLM provider. Default is keyless mode."""
        provider = (self.llm_provider or "none").lower()

        if provider in ("none", "off", "disabled", "keyless"):
            self.llm_provider = "none"
            self.llm = None
            logger.info("RAG LLM provider disabled; using keyless grounded response mode")
            return

        if provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                self.llm_provider = "none"
                self.llm = None
                logger.warning("Gemini provider requested but no GOOGLE_API_KEY/GEMINI_API_KEY found; using keyless mode")
                return

            try:
                GoogleGenerativeAI = _import_symbol(
                    ["langchain_google_genai"],
                    "GoogleGenerativeAI",
                )
                self.llm = GoogleGenerativeAI(
                    model=self.model,
                    temperature=0.7,
                    max_tokens=1024,
                    google_api_key=api_key,
                )
                return
            except Exception as e:
                self.llm_provider = "none"
                self.llm = None
                logger.warning("Failed to initialize Gemini provider (%s); using keyless mode", e)
                return

        self.llm_provider = "none"
        self.llm = None
        logger.warning("Unknown RAG_LLM_PROVIDER=%s; using keyless mode", provider)

    def _generate_keyless_response(self, query: str, relevant_docs: List[Any]) -> str:
        """Return a grounded response without external LLM APIs."""
        snippets: List[str] = []
        for idx, doc in enumerate(relevant_docs[:4], start=1):
            content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
            if not content:
                continue
            trimmed = content[:260] + ("..." if len(content) > 260 else "")
            source = getattr(doc, "metadata", {}).get("source", "knowledge_base")
            snippets.append(f"{idx}. ({source}) {trimmed}")

        if not snippets:
            return (
                "I could not find enough grounded information in the local index for that query. "
                "Please index more data and try again."
            )

        return (
            f"Grounded answer (keyless mode) for: '{query}'\n\n"
            "Relevant evidence:\n"
            + "\n".join(snippets)
            + "\n\n"
            "Tip: Set RAG_LLM_PROVIDER=gemini with a valid API key if you want generated narrative answers."
        )

    def index_universities_database(self, json_path: str):
        """
        Index the universities database.

        Args:
            json_path: Path to the universities JSON database
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []

            # Process each country
            for country, universities in data.items():
                if country == "metadata":
                    continue

                for uni in universities:
                    # Create comprehensive text representation
                    text = f"""
                    University: {uni['name']}
                    Country: {uni['country']}
                    City: {uni['city']}
                    Rankings: QS World #{uni['rankings'].get('qs_world', 'N/A')}, THE World #{uni['rankings'].get('the_world', 'N/A')}
                    Programs: {', '.join(uni['programs'])}
                    Tuition: Undergraduate £{uni['tuition']['undergraduate_intl_gbp']}, Postgraduate £{uni['tuition']['postgraduate_intl_gbp']}
                    Requirements: GPA {uni['acceptance_criteria']['min_grade_point']}, IELTS {uni['acceptance_criteria']['ielts_min']}, TOEFL {uni['acceptance_criteria']['toefl_min']}
                    Website: {uni['website']}
                    Interview Required: {uni['interview_required']}
                    Additional Tests: {', '.join(uni.get('additional_tests', []))}
                    """

                    doc = self.DocumentCls(
                        page_content=text.strip(),
                        metadata={
                            "source": "universities_database",
                            "university_id": uni['id'],
                            "country": uni['country'],
                            "type": "university_info"
                        }
                    )
                    documents.append(doc)

            # Split and add to vectorstore
            split_docs = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(split_docs)

            logger.info(f"Indexed {len(documents)} university documents")

        except Exception as e:
            logger.error(f"Error indexing universities database: {e}")

    def index_scraped_data(self, scraped_dir: str):
        """
        Index scraped web data.

        Args:
            scraped_dir: Directory containing scraped data files
        """
        scraped_path = Path(scraped_dir)
        if not scraped_path.exists():
            logger.warning(f"Scraped data directory {scraped_dir} does not exist")
            return

        documents = []

        # Process all text files in scraped directory
        for file_path in scraped_path.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                doc = self.DocumentCls(
                    page_content=content,
                    metadata={
                        "source": "scraped_data",
                        "file_path": str(file_path),
                        "type": "web_content"
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        # Split and add to vectorstore
        split_docs = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(split_docs)

        logger.info(f"Indexed {len(documents)} scraped documents")

    def index_pdf_documents(self, pdf_dir: str):
        """
        Index PDF documents.

        Args:
            pdf_dir: Directory containing PDF files
        """
        try:
            PyPDFLoader = _import_symbol(
                ["langchain_community.document_loaders"],
                "PyPDFLoader",
            )

            pdf_path = Path(pdf_dir)
            if not pdf_path.exists():
                logger.warning(f"PDF directory {pdf_dir} does not exist")
                return

            documents = []

            for pdf_file in pdf_path.rglob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": "pdf_documents",
                            "file_path": str(pdf_file),
                            "type": "pdf_content"
                        })
                    documents.extend(docs)

                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file}: {e}")

            # Split and add to vectorstore
            split_docs = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(split_docs)

            logger.info(f"Indexed {len(documents)} PDF documents")

        except ImportError:
            logger.warning("PyPDFLoader not available. Install pypdf to index PDF documents")

    def retrieve_relevant_info(self, query: str, k: Optional[int] = None) -> List[Any]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        try:
            if k is None:
                k = self.default_top_k
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        relevant_docs: Optional[List[Any]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG and Gemini.

        Args:
            query: User query
            context: Additional context from the chatbot

        Returns:
            Dict with response and source documents
        """
        try:
            if relevant_docs is None:
                relevant_docs = self.retrieve_relevant_info(query=query, k=k)

            # Build context from retrieved documents
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Create augmented prompt
            system_prompt = """
            You are a helpful study abroad assistant. Use the provided context to give accurate,
            personalized advice about universities, eligibility, costs, and applications.

            Context from knowledge base:
            {context}

            User query: {query}

            Additional context: {additional_context}

            Provide a helpful, empathetic response based on the context above.
            """

            additional_context = ""
            if context:
                additional_context = f"User profile: {context.get('profile_data', {})}"

            full_prompt = system_prompt.format(
                context=context_text,
                query=query,
                additional_context=additional_context
            )

            # Generate response using selected provider; keyless mode is default.
            if self.llm is None:
                text = self._generate_keyless_response(query=query, relevant_docs=relevant_docs)
            else:
                response = self.llm.invoke(full_prompt)
                text = response.content if hasattr(response, 'content') else str(response)

            return {
                "response": text,
                "source_documents": relevant_docs,
                "query": query,
                "llm_provider": self.llm_provider,
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "source_documents": [],
                "query": query
            }

    def answer_with_context(self, query: str, context: Optional[Dict[str, Any]] = None, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve supporting context and generate a grounded response.

        Args:
            query: User question.
            context: Optional user/session context.
            k: Number of documents to retrieve.

        Returns:
            Response payload with text and source snippets.
        """
        try:
            if k is None:
                k = self.default_top_k
            relevant_docs = self.retrieve_relevant_info(query=query, k=k)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            if not context_text.strip():
                return {
                    "response": "I don't have enough indexed knowledge yet. Please add data to the vector index and try again.",
                    "source_documents": [],
                    "query": query,
                }

            response = self.generate_response(query=query, context=context, relevant_docs=relevant_docs, k=k)
            response["source_count"] = len(relevant_docs)
            return response
        except Exception as e:
            logger.error("Error in answer_with_context: %s", e)
            return {
                "response": "I couldn't complete a RAG lookup right now. Please try again shortly.",
                "source_documents": [],
                "query": query,
                "source_count": 0,
            }

    def persist_vectorstore(self):
        """Persist the vector store to disk."""
        try:
            self.vectorstore.persist()
            logger.info("Vector store persisted")
        except Exception as e:
            logger.error(f"Error persisting vector store: {e}")

    def load_existing_index(self):
        """Load existing vector store if available."""
        try:
            if self.persist_directory.exists() and any(self.persist_directory.iterdir()):
                logger.info("Loading existing vector store")
                self.vectorstore = self.ChromaCls(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embedding_model
                )
        except Exception as e:
            logger.error(f"Error loading existing vector store: {e}")