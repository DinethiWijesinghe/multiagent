"""
Retrieval-Augmented Generation (RAG) System
===========================================
Integrates external knowledge sources with Gemini 2.5 Flash for improved NLP accuracy.

Features:
- Indexes documents, databases, and web content
- Retrieves relevant information based on user queries
- Augments prompts sent to Gemini for context-aware responses
- Supports multiple data sources: JSON databases, PDFs, text files

Components:
- VectorStore: ChromaDB for document storage and retrieval
- Embeddings: Sentence Transformers for text vectorization
- Retriever: Semantic search with similarity scoring
- Generator: Gemini 2.5 Flash for response generation
"""

# pyright: reportMissingImports=false

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import importlib

from langchain_google_genai import GoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Initialize Gemini model
        self.llm = GoogleGenerativeAI(
            model=self.model,
            temperature=0.7,
            max_tokens=1024
        )

        # Initialize text splitter
        self.text_splitter = self.TextSplitterCls(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        logger.info("RAG system initialized with model %s", self.model)

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

    def retrieve_relevant_info(self, query: str, k: int = 3) -> List[Any]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using RAG and Gemini.

        Args:
            query: User query
            context: Additional context from the chatbot

        Returns:
            Dict with response and source documents
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_info(query)

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

            # Generate response using Gemini
            response = self.llm.invoke(full_prompt)

            return {
                "response": response.content if hasattr(response, 'content') else str(response),
                "source_documents": relevant_docs,
                "query": query
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "source_documents": [],
                "query": query
            }

    def answer_with_context(self, query: str, context: Optional[Dict[str, Any]] = None, k: int = 4) -> Dict[str, Any]:
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
            relevant_docs = self.retrieve_relevant_info(query=query, k=k)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            if not context_text.strip():
                return {
                    "response": "I don't have enough indexed knowledge yet. Please add data to the vector index and try again.",
                    "source_documents": [],
                    "query": query,
                }

            response = self.generate_response(query=query, context=context)
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