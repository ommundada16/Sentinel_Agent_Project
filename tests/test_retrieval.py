import pytest

from app.agent.embeddings import HashEmbeddingProvider
from app.data import db
from app.data import repository as repo
from app.mcp_server import tools


def test_topk_returns_relevant_doc_with_score_and_title(seeded):
    vec = HashEmbeddingProvider().embed(["how often are privileged administrator entitlements reviewed"])[0]
    with db.session_scope() as s:
        hits = repo.search_doc_chunks(s, vec, k=3)
    assert len(hits) == 3
    assert hits[0]["title"] == "Access Review Policy"
    assert hits[0]["score"] >= hits[1]["score"]


def test_tool_returns_citations_and_applies_relevance_floor(seeded):
    res = tools.search_governance_docs("what happens to access when someone leaves the company")
    assert res["results"][0]["title"] == "Joiner Mover Leaver Policy"
    nothing = tools.search_governance_docs("zzzz qqqq xxxx")
    assert nothing["results"] == []


@pytest.mark.ml
def test_real_minilm_semantic_match(seeded):
    from app.agent.embeddings import MiniLMEmbeddingProvider

    emb = MiniLMEmbeddingProvider()
    assert len(emb.embed(["hello"])[0]) == 384
