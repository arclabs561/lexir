# lexir

[![crates.io](https://img.shields.io/crates/v/lexir.svg)](https://crates.io/crates/lexir)
[![Documentation](https://docs.rs/lexir/badge.svg)](https://docs.rs/lexir)
[![CI](https://github.com/arclabs561/lexir/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/lexir/actions/workflows/ci.yml)

Lexical IR on top of postings lists.

**Status**: experimental. This repository is public as a reference implementation; it is not
currently packaged for `crates.io`.

## Feature Selection

- **`default`**: Includes `persistence`.
- **In-memory only**: disable default features.

## What it is

`lexir` is the scoring/ranking layer. Candidate generation and storage live in `postings`.

## Building

`lexir` is not on `crates.io` yet; use a git dependency:

```toml
[dependencies]
lexir = { git = "https://github.com/arclabs561/lexir" }
```

Notes:

- `postings`, `rankfns`, and `durability` are pulled as git dependencies.
- `gramdex` and `textprep` are pulled from `crates.io` (and are only used when their features are enabled).

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

## Features

- `persistence` (default): save/load via `durability` + `postings/persistence`
- `recordlog`: append-only operation logs for rebuildable indexes (CLI uses this)
- `cli`: enables the `lexir` CLI (debugging + end-to-end validation)
- `fuzzy`: fuzzy query expansion via `gramdex` — expands only **OOV terms** (terms not in the index); in-vocabulary terms are used as-is

## CLI (with `--features cli`)

```bash
cargo run --features cli -- <subcommand>
```

**Indexing & search**: `index`, `search-index`, `search` (build/search from corpus or saved index).

**Record-log operations** (append-only ops + checkpoint):
- `log-add`, `log-delete`, `log-search` — incremental updates and search over log
- `log-checkpoint`, `log-compact`, `log-status` — checkpoint management
- `log-doctor --root <dir> [--fix]` — repair missing meta files
- `log-prune --root <dir>` — prune redundant checkpoints
- `log-scan --root <dir> [--strict]` — validate record log integrity
- `log-validate --root <dir>` — verify checkpoint + log consistency
- `log-serve` — serve search over a log-backed index
