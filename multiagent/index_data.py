"""
Data Indexing Script for RAG System
====================================
Indexes all available data sources for the RAG system.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.rag_system import RAGSystem


def main():
    """Index all available data sources."""

    # Initialize RAG system
    rag = RAGSystem()

    # Index universities database
    universities_path = project_root / "data" / "databases" / "universities_database.json"
    if universities_path.exists():
        print("Indexing universities database...")
        rag.index_universities_database(str(universities_path))
    else:
        print(f"Universities database not found at {universities_path}")

    # Index training data
    training_path = project_root / "data" / "training" / "eligibility_training_data.json"
    if training_path.exists():
        print("Indexing training data...")
        # For now, we'll treat the training data as a general knowledge document
        with open(training_path, 'r', encoding='utf-8') as f:
            content = f.read()

        doc = rag.DocumentCls(
            page_content=content,
            metadata={
                "source": "training_data",
                "type": "eligibility_requirements"
            }
        )
        split_docs = rag.text_splitter.split_documents([doc])
        rag.vectorstore.add_documents(split_docs)
        print(f"Indexed training data")
    else:
        print(f"Training data not found at {training_path}")

    # Index scraped data (if any)
    scraped_dir = project_root / "data" / "scraped"
    if scraped_dir.exists():
        print("Indexing scraped data...")
        rag.index_scraped_data(str(scraped_dir))
    else:
        print(f"Scraped data directory not found at {scraped_dir}")

    # Index any PDF documents in data folder
    data_dir = project_root / "data"
    print("Indexing PDF documents...")
    rag.index_pdf_documents(str(data_dir))

    # Persist the vector store
    print("Persisting vector store...")
    rag.persist_vectorstore()

    print("Data indexing complete!")


if __name__ == "__main__":
    main()