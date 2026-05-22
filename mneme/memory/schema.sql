-- Mneme memory schema. One SQLite file holds everything.
-- Applied once by store.init_db(). See docs/MEMORY.md.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Immutable text units: user messages, assistant replies, extracted facts.
CREATE TABLE IF NOT EXISTS slices (
    id          TEXT PRIMARY KEY,            -- ULID
    role        TEXT NOT NULL,               -- 'user' | 'assistant' | 'fact'
    text        TEXT NOT NULL,
    turn_id     TEXT,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slices_turn ON slices(turn_id);

-- Concepts: people, artifacts, ideas, time references. Deduplicated by name.
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,            -- ULID
    name        TEXT NOT NULL UNIQUE,        -- canonical, normalized
    kind        TEXT NOT NULL,               -- concept|person|time|artifact
    first_seen  INTEGER NOT NULL
);

-- Typed, directed relations. slice_id records which slice introduced the edge.
CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,            -- ULID
    src         TEXT NOT NULL REFERENCES nodes(id),
    dst         TEXT NOT NULL REFERENCES nodes(id),
    type        TEXT NOT NULL,               -- ABOUT|MENTIONS|REFERENCES|ALIAS|BEFORE|AFTER|PART_OF
    slice_id    TEXT NOT NULL REFERENCES slices(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src   ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst   ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_slice ON edges(slice_id);

-- Embedding cache for arbitrary text (queries, repeated phrases, slices).
CREATE TABLE IF NOT EXISTS embeddings_cache (
    text_hash   TEXT PRIMARY KEY,            -- sha256(text)[:16]
    vector      BLOB NOT NULL,
    created_at  INTEGER NOT NULL
);

-- Searchable vectors (sqlite-vec virtual table), 1:1 with slices.
-- Created separately by store.init_db() after the vec0 extension is loaded:
--   CREATE VIRTUAL TABLE IF NOT EXISTS vec_slices USING vec0(
--       slice_id TEXT PRIMARY KEY, embedding FLOAT[768]);
