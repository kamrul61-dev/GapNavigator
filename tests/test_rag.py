import os
import pytest
from langchain_core.embeddings import FakeEmbeddings
from app.rag import retriever

def test_rag_pipeline_indexing_and_query(monkeypatch, tmp_path):
    """
    Verifies that the RAG pipeline can split markdown document resources,
    index them in a FAISS vector store, save/load from disk, and search.
    
    Uses FakeEmbeddings to run offline and isolates files in a temp directory.
    """
    # 1. Setup Mock Embeddings
    fake_embeddings = FakeEmbeddings(size=768)
    monkeypatch.setattr(retriever, "get_embeddings_model", lambda: fake_embeddings)
    
    # 2. Setup Temporary Directories
    temp_index = tmp_path / "temp_faiss_index"
    temp_kb = tmp_path / "kb"
    temp_kb.mkdir()
    
    # Write a mock resource file
    mock_resources = """
# Domain: Data Science
- **Title**: Intro to Pandas
  **URL**: https://pandas-test.org
  **Type**: Tutorial
  **Description**: Learn Pandas DataFrame operations and indexing.
  **Skills**: Pandas, Python

- **Title**: Advanced SQL Window Functions
  **URL**: https://sql-test.org
  **Type**: Course
  **Description**: Learn window functions and CTE expressions.
  **Skills**: SQL, Databases
    """
    with open(temp_kb / "resources.md", "w", encoding="utf-8") as f:
        f.write(mock_resources)
        
    monkeypatch.setattr(retriever, "KB_DIR", str(temp_kb))
    monkeypatch.setattr(retriever, "INDEX_PATH", str(temp_index))
    
    # 3. Build Vector Store
    db = retriever.build_vector_store()
    assert db is not None
    assert os.path.exists(os.path.join(str(temp_index), "index.faiss"))
    
    # 4. Test Retrieval
    # Since FakeEmbeddings maps all texts to the same vector, similarity search
    # will find both at distance 0. We verify that both are in the index.
    retrieved = retriever.retrieve_resources("Pandas DataFrame", k=2)
    assert len(retrieved) == 2
    
    titles = [doc for doc in retrieved]
    assert any("Intro to Pandas" in t for t in titles)
    assert any("Advanced SQL Window Functions" in t for t in titles)
    assert any("https://pandas-test.org" in t for t in titles)
    assert any("https://sql-test.org" in t for t in titles)
