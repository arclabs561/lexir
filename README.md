# lexir

[![crates.io](https://img.shields.io/crates/v/lexir.svg)](https://crates.io/crates/lexir)
[![Documentation](https://docs.rs/lexir/badge.svg)](https://docs.rs/lexir)
[![CI](https://github.com/arclabs561/lexir/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/lexir/actions/workflows/ci.yml)

Lexical IR on top of postings lists.

**Status**: experimental. Published on `crates.io`; the API may still shift.

## Feature Selection

- **`default`**: Includes `persistence`.
- **In-memory only**: disable default features.

## What it is

`lexir` is the scoring/ranking layer. Candidate generation and storage live in `postings`.

## Building

Add from `crates.io`:

```toml
[dependencies]
lexir = "0.2"
```

Notes:

- Optional persistence, CLI, and fuzzy-search dependencies are enabled only by
  their feature flags.
- The `fuzzy` feature uses `gramdex` for k-gram candidate generation.

## Usage (library)

**BM25** (default):

```rust
use lexir::bm25::{Bm25Params, InvertedIndex};

let mut idx = InvertedIndex::new();
idx.add_document(1, &["hello".to_string(), "world".to_string()]);
let hits = idx.retrieve(&["hello".to_string()], 10, Default::default()).unwrap();
assert_eq!(hits[0].0, 1);
```

**TF-IDF** (requires multiple docs for non-zero IDF):

```rust
use lexir::tfidf::{TfIdfParams, retrieve_tfidf};
use lexir::bm25::InvertedIndex;

let mut idx = InvertedIndex::new();
idx.add_document(1, &["hello".to_string(), "world".to_string()]);
idx.add_document(2, &["other".to_string()]);  // IDF(hello) > 0
let hits = retrieve_tfidf(&idx, &["hello".to_string()], 10, TfIdfParams::linear()).unwrap();
assert_eq!(hits[0].0, 1);
```

**Query Likelihood** (Dirichlet \(\mu=1000\) via `QueryLikelihoodParams::default()`):

```rust
use lexir::query_likelihood::{QueryLikelihoodParams, retrieve_query_likelihood};
use lexir::bm25::InvertedIndex;

let mut idx = InvertedIndex::new();
idx.add_document(1, &["hello".to_string(), "world".to_string()]);
let hits = retrieve_query_likelihood(&idx, &["hello".to_string()], 10, QueryLikelihoodParams::default()).unwrap();
assert!(!hits.is_empty());
```

## Examples

Runnable examples live in [`examples/`](examples/), including `readme_examples`,
which exercises the snippets shown above, and `raw_bm25_segment`, which scores a
file-backed `postings::raw::RawSegmentFile`.

## File-backed BM25

Enable `raw-segment` to score numeric postings segments without opening a full
in-memory `InvertedIndex`:

```toml
[dependencies]
lexir = { version = "0.2", features = ["raw-segment"] }
```

`lexir::raw::retrieve_bm25_raw_file` takes a mutable
`postings::raw::RawSegmentFile`, query term ids, `k`, and `Bm25Params`. The
caller owns the lexicon, commit lifecycle, deletes, and segment merge policy.

## Features

- `persistence` (default): save/load via `durability` + `postings/persistence`
- `recordlog`: append-only operation logs for rebuildable indexes (CLI uses this)
- `cli`: enables the `lexir` CLI (debugging + end-to-end validation)
- `fuzzy`: fuzzy query expansion via `gramdex` — expands only **OOV terms** (terms not in the index); in-vocabulary terms are used as-is
- `raw-segment`: BM25 scoring over `postings::raw` byte- and file-backed segments

## CLI (with `--features cli`)

```bash
cargo run --features cli -- <subcommand>
```

Subcommands: `index`, `search-index`, `search` for indexing and search. Record-log operations: `log-add`, `log-delete`, `log-search`, `log-checkpoint`, `log-compact`, `log-status`, `log-doctor`, `log-prune`, `log-scan`, `log-validate`, `log-serve`.
