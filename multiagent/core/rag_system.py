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
import re
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
KEYLESS_EVIDENCE_LIMIT = int(os.environ.get("KEYLESS_EVIDENCE_LIMIT", "6"))
KEYLESS_SNIPPET_CHARS = int(os.environ.get("KEYLESS_SNIPPET_CHARS", "420"))


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
                ChatGoogleGenerativeAI = _import_symbol(
                    ["langchain_google_genai"],
                    "ChatGoogleGenerativeAI",
                )
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=0.7,
                    max_output_tokens=1024,
                    google_api_key=api_key,
                )
                logger.info("Gemini provider initialised with model %s", self.model)
                return
            except Exception as e:
                self.llm_provider = "none"
                self.llm = None
                logger.warning("Failed to initialize Gemini provider (%s); using keyless mode", e)
                return

        self.llm_provider = "none"
        self.llm = None
        logger.warning("Unknown RAG_LLM_PROVIDER=%s; using keyless mode", provider)

    def _build_chat_messages(self, query: str, context_text: str, context: Optional[Dict[str, Any]], conversation_history: List[Dict[str, Any]]) -> Any:
        """Build a LangChain message list for multi-turn Gemini calls."""
        try:
            SystemMessage = _import_symbol(["langchain_core.messages"], "SystemMessage")
            HumanMessage = _import_symbol(["langchain_core.messages"], "HumanMessage")
            AIMessage = _import_symbol(["langchain_core.messages"], "AIMessage")
        except ImportError:
            # Fallback: single-string prompt for completion-style LLMs
            profile_str = ""
            if context:
                profile = context.get("profile_data", {})
                if profile:
                    profile_str = f"\nUser profile: {json.dumps(profile, ensure_ascii=False)}"
            return f"Knowledge base:\n{context_text}{profile_str}\n\nQuestion: {query}\nAnswer:"

        profile_lines = ""
        if context:
            profile = context.get("profile_data") or {}
            if profile:
                profile_lines = (
                    "\n\nUser profile:\n"
                    + "\n".join(f"  {k}: {v}" for k, v in profile.items() if v is not None)
                )

        agent_results_lines = ""
        if context:
            agent_results = context.get("agent_results") or {}
            if agent_results:
                parts = []
                for key, value in agent_results.items():
                    if value and isinstance(value, dict):
                        snippet = json.dumps(value, ensure_ascii=False)
                        parts.append(f"  {key}: {snippet[:600]}")
                if parts:
                    agent_results_lines = (
                        "\n\nSpecialized agent analysis (use this to personalise your answer):\n"
                        + "\n".join(parts)
                    )

        external_factor_lines = ""
        if context:
            external_factors = context.get("external_factors") or []
            if external_factors:
                labels = [
                    factor.get("label", "")
                    for factor in external_factors
                    if isinstance(factor, dict) and factor.get("label")
                ]
                if labels:
                    external_factor_lines = (
                        "\n\nPriority external factors to address in the answer:\n"
                        + "\n".join(f"  - {label}" for label in labels[:6])
                    )

        system_content = (
            "You are UniAssist, a helpful and empathetic study-abroad advisor for Sri Lankan students. "
            "Use the knowledge base context below to give accurate, personalised answers about "
            "university eligibility, tuition costs, scholarships, visa requirements, and applications. "
            "If the context doesn't cover the question, say so honestly rather than guessing. "
            "When external factors are provided, explicitly connect your advice to them and end with practical next steps."
            "\n\nKnowledge base context:\n" + context_text
            + profile_lines
            + agent_results_lines
            + external_factor_lines
        )

        messages: list = [SystemMessage(content=system_content)]

        # Inject last 10 conversation turns for continuity
        for turn in (conversation_history or [])[-10:]:
            role = (turn.get("role") or "").lower()
            text = (turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role in ("assistant", "bot", "ai"):
                messages.append(AIMessage(content=text))

        messages.append(HumanMessage(content=query))
        return messages

    def _generate_keyless_response(self, query: str, relevant_docs: List[Any]) -> str:
        """Return a grounded response without external LLM APIs."""
        snippets: List[str] = []
        snippet_limit = max(2, KEYLESS_EVIDENCE_LIMIT)
        snippet_chars = max(160, KEYLESS_SNIPPET_CHARS)

        for idx, doc in enumerate(relevant_docs[:snippet_limit], start=1):
            content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
            if not content:
                continue
            trimmed = content[:snippet_chars] + ("..." if len(content) > snippet_chars else "")
            source = getattr(doc, "metadata", {}).get("source", "knowledge_base")
            snippets.append(f"{idx}. ({source}) {trimmed}")

        if not snippets:
            return (
                "I could not find enough grounded information in the local index for that query. "
                "Please index more data and try again."
            )

        q = (query or "").lower()
        next_steps: List[str] = []
        if any(word in q for word in ["deadline", "intake", "date", "timeline"]):
            next_steps.append("Create an application timeline and prioritize the earliest intake deadlines.")
        if any(word in q for word in ["visa", "immigration", "permit"]):
            next_steps.append("Compile visa documents early and verify requirements from official embassy sources.")
        if any(word in q for word in ["cost", "budget", "tuition", "scholarship", "fee"]):
            next_steps.append("Compare tuition and living costs across options and shortlist scholarship-eligible programs.")
        if any(word in q for word in ["eligibility", "requirement", "gpa", "ielts", "toefl", "pte"]):
            next_steps.append("Check your GPA and language scores against each program minimum before applying.")
        if not next_steps:
            next_steps.append("Review the cited evidence and confirm details on each university's official admissions page.")

        return (
            f"Grounded answer (keyless mode) for: '{query}'\n\n"
            "Relevant evidence:\n"
            + "\n".join(snippets)
            + "\n\n"
            + "Suggested next steps:\n"
            + "\n".join(f"- {step}" for step in next_steps[:3])
            + "\n\n"
            "Tip: Set RAG_LLM_PROVIDER=gemini with a valid API key if you want generated narrative answers."
        )

    def _extract_relevant_urls(self, relevant_docs: List[Any], limit: int = 5) -> List[str]:
        """Extract unique relevant URLs from retrieved documents."""
        urls: List[str] = []
        seen = set()

        for doc in relevant_docs or []:
            metadata = getattr(doc, "metadata", {}) or {}

            # Prefer explicit URL metadata if present.
            for key in ("website", "url", "source_url"):
                value = metadata.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")) and value not in seen:
                    seen.add(value)
                    urls.append(value)
                    if len(urls) >= limit:
                        return urls

            # Fallback: parse URLs from the text body.
            body = getattr(doc, "page_content", "") or ""
            matches = re.findall(r"https?://[^\s)\]>\"']+", body)
            for url in matches:
                cleaned = url.rstrip(".,;")
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    urls.append(cleaned)
                    if len(urls) >= limit:
                        return urls

        return urls

    def _extract_question_options(self, query: str, relevant_docs: List[Any], limit: int = 4) -> List[Dict[str, str]]:
        """Build concise option cards from retrieved docs for UI-friendly follow-up choices."""
        options: List[Dict[str, str]] = []
        seen_names = set()

        # Query-aware ranking: score option candidates by query term overlap.
        stop_terms = {
            "the", "and", "for", "with", "from", "what", "which", "about", "please", "show",
            "tell", "me", "want", "need", "university", "universities", "college", "best",
        }
        query_terms = [
            t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(t) >= 3 and t not in stop_terms
        ]
        candidates: List[tuple[float, Dict[str, str]]] = []

        for doc in relevant_docs or []:
            body = (getattr(doc, "page_content", "") or "").strip()
            if not body:
                continue

            metadata = getattr(doc, "metadata", {}) or {}

            name_match = re.search(r"University:\s*(.+)", body, flags=re.IGNORECASE)
            country_match = re.search(r"Country:\s*(.+)", body, flags=re.IGNORECASE)
            website_match = re.search(r"Website:\s*(https?://\S+)", body, flags=re.IGNORECASE)

            name = (name_match.group(1).strip() if name_match else metadata.get("university_id") or "Option").strip()
            if name in seen_names:
                continue

            country = (country_match.group(1).strip() if country_match else metadata.get("country") or "").strip()
            website = ""
            if website_match:
                website = website_match.group(1).rstrip(".,;")
            elif isinstance(metadata.get("website"), str):
                website = metadata.get("website", "")

            option = {
                "name": name,
                "country": country,
                "website": website,
            }

            # Build a searchable text for query-aware ranking.
            searchable = " ".join([
                name.lower(),
                country.lower(),
                body[:700].lower(),
            ])
            score = 0.0
            if query_terms:
                for term in query_terms:
                    if term in searchable:
                        score += 1.0
                # Prioritize explicit country matches.
                if country and any(term == country.lower() for term in query_terms):
                    score += 1.5
            else:
                # Preserve deterministic ordering when query has no informative terms.
                score = 0.1

            # Slight preference for options with a website link.
            if website:
                score += 0.05

            candidates.append((score, option))
            seen_names.add(name)

        # Highest query match first, then name for deterministic ties.
        candidates.sort(key=lambda item: (-item[0], item[1].get("name", "")))
        for _, option in candidates:
            options.append(option)
            if len(options) >= limit:
                break

        return options

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
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG and Gemini (multi-turn).

        Args:
            query: User query
            context: Additional context (user profile, etc.)
            relevant_docs: Pre-retrieved docs (fetched if None)
            k: Retrieval top-k override
            conversation_history: List of prior {role, text} dicts for multi-turn continuity

        Returns:
            Dict with response and source documents
        """
        try:
            if relevant_docs is None:
                relevant_docs = self.retrieve_relevant_info(query=query, k=k)

            relevant_urls = self._extract_relevant_urls(relevant_docs=relevant_docs)
            options = self._extract_question_options(query=query, relevant_docs=relevant_docs)

            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Generate response using selected provider; keyless mode is default.
            if self.llm is None:
                text = self._generate_keyless_response(query=query, relevant_docs=relevant_docs)
            else:
                messages = self._build_chat_messages(
                    query=query,
                    context_text=context_text,
                    context=context,
                    conversation_history=conversation_history or [],
                )
                response = self.llm.invoke(messages)
                text = response.content if hasattr(response, "content") else str(response)

            return {
                "response": text,
                "source_documents": relevant_docs,
                "query": query,
                "llm_provider": self.llm_provider,
                "relevant_urls": relevant_urls,
                "options": options,
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "source_documents": [],
                "query": query,
                "relevant_urls": [],
                "options": [],
            }

    def answer_with_context(self, query: str, context: Optional[Dict[str, Any]] = None, k: Optional[int] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Retrieve supporting context and generate a grounded response.

        Args:
            query: User question.
            context: Optional user/session context (profile_data, etc.).
            k: Number of documents to retrieve.
            conversation_history: Prior {role, text} turns for multi-turn continuity.

        Returns:
            Response payload with text and source snippets.
        """
        try:
            if k is None:
                k = self.default_top_k
            relevant_docs = self.retrieve_relevant_info(query=query, k=k)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            if not context_text.strip() and self.llm is None:
                return {
                    "response": "I don't have enough indexed knowledge yet. Please add data to the vector index and try again.",
                    "source_documents": [],
                    "query": query,
                }

            response = self.generate_response(
                query=query,
                context=context,
                relevant_docs=relevant_docs,
                k=k,
                conversation_history=conversation_history,
            )
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