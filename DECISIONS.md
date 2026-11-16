# Design decisions

Each entry: what we chose, the alternative, and why.

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| 1 | Hand-written agent loop (`app/agent/core.py`) | LangChain / LangGraph | Every step is visible and explainable; simple composable patterns beat hidden abstraction. |
| 2 | Tools served by an MCP server over stdio | Direct Python calls | Decouples capabilities from reasoning; a reviewer can audit tools alone. |
| 3 | LLM Strategy (`providers.py`): Anthropic, Ollama, FakeLLM | One hardcoded SDK | No lock-in; deterministic, free tests and evals via FakeLLM. |
| 4 | Pydantic models are the single source for LLM tool schemas AND validation (`extra="forbid"`) | Trusting raw model JSON | Hallucinated/malformed calls are rejected and the error is fed back so the model can self-correct. |
| 5 | Access decisions in deterministic code (`check_policy`) | Letting the LLM decide | Correct, explainable, testable. The model surfaces the decision, never makes it. |
| 6 | Approval gate on every WRITE, default-deny, atomic resolve | Console prompt / fail-open | Governance must fail closed. Only an exact `approve` runs the action; a conditional `UPDATE` stops double approval. |
| 7 | Injection defense = delimiters + sentence-level quarantine + gate on writes | Detector only | Defense in depth: even a fully fooled model cannot write without a human (proven in `test_injected_document_is_quarantined_and_write_still_gated` and eval tasks t08-t10). |
| 8 | PostgreSQL + pgvector + HNSW, SQLAlchemy repository | Separate vector DB; in-memory cosine | One store, real ANN index, honest scale story. SQLite is a dev/test fallback only (brute-force cosine in Python). |
| 9 | Chunks of 150 words with 25 overlap | ~400-token chunks | MiniLM truncates at 256 tokens; bigger chunks would be silently cut (known pitfall from the Auditor project). |
| 10 | Same embedder for docs and queries via `EmbeddingProvider`; hash embedder for tests | Mixed/real model in tests | Mismatched embedders silently break retrieval; the fake keeps tests offline. |
| 11 | Bounded everything: `MAX_STEPS`, `LIMIT`, pagination + `more_available`, 4,000-char tool results | Dump data into the prompt | Correct at scale; no runaway loops; no context blowups. |
| 12 | Stateless API; conversation + pending approvals stored in DB | In-process dicts | Horizontally scalable; approvals survive restarts. |
| 13 | Audit rows written in their own transaction, emails masked, secrets redacted | `print()` / verbatim logs | Auditability without leaking PII; tokens, latency and cost per call. |
| 14 | Revocation = a `revoked` access_request row | Deleting role links | Roles are shared by many users; revoke must affect one user and stay auditable. |
| 15 | Approval required for every WRITE (not for READs on high-sensitivity entitlements) | Spec's "or sensitivity=high" clause | All writes are already gated; gating `check_policy` would make a human click just to ask a question. |
| 16 | DB tests: delete+reseed per test, not rollback | Transaction rollback | Tools commit in their own sessions exactly like production. |
| 17 | `alembic` 0001 delegates to `init_schema` | Hand-written DDL | One schema definition shared with tests; later migrations should be real diffs. |
| 18 | Evals use scripted "compromised" models for injection tasks | Only well-behaved scripts | Measures OUR defenses assuming the model is fooled. Tool-selection accuracy excludes injection tasks (they attempt writes by design). |

## Known limits / next steps
- Injection detection is pattern-based; a determined attacker can rephrase. The approval gate is the real backstop.
- The hash embedder is lexical, not semantic; use `EMBEDDING_PROVIDER=minilm` for real semantics.
- Next: real IdP integration, API auth/RBAC, rate limits, a re-ranker, an audit dashboard.
