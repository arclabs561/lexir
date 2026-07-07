# lexir

[![crates.io](https://img.shields.io/crates/v/lexir.svg)](https://crates.io/crates/lexir)
[![Documentation](https://docs.rs/lexir/badge.svg)](https://docs.rs/lexir)

Lexical scoring over postings lists.

**Status**: experimental. Published on `crates.io`; the API may still shift.

## What it is

`lexir` is the scoring/ranking layer. Candidate generation and storage live in `postings`.
Exact filters such as phrase/proximity matching should produce candidate doc ids,
then pass those candidates to `lexir` for ranking.

## Storage model

`InvertedIndex` is the in-memory path: postings and corpus statistics live in
RAM, with optional save/load support through `persistence`.

The `raw-segment` feature is the file-backed path for larger immutable segment
sets. `lexir` scores `postings::raw` segment files directly, keeps only fixed
segment directories and BM25 statistics in memory, and reads posting payloads
from the segment files during search. Multi-segment raw search treats the files
as one corpus by sharing corpus-level IDF and average document length, then
merges per-segment top-k results with a conservative segment-pruning bound.

`lexir` does not own tokenization, term-id assignment, positional storage,
phrase/proximity execution, raw segment commit lifecycle, deletes, or segment
merges. Those stay with `postings`, the caller, or a storage layer above
`postings::raw`.

## Building

```toml
[dependencies]
lexir = "0.2"
```

## Usage (library)

**BM25** (default):

```rust
use lexir::bm25::{Bm25Params, InvertedIndex};

let mut idx = InvertedIndex::new();
idx.add_document(1, &["hello".to_string(), "world".to_string()]);
let hits = idx.retrieve(&["hello".to_string()], 10, Default::default()).unwrap();
assert_eq!(hits[0].0, 1);
```

**BM25 over prefiltered candidates**:

```rust
use lexir::bm25::{Bm25Params, InvertedIndex};

let mut idx = InvertedIndex::new();
idx.add_document(1, &["quick".to_string(), "brown".to_string(), "fox".to_string()]);
idx.add_document(2, &["quick".to_string(), "brown".to_string(), "dog".to_string()]);

// Candidate ids can come from `postings::positional`, a boolean planner, or caller code.
let phrase_candidates = [2];
let hits = idx
    .retrieve_candidates(
        &["quick".to_string(), "brown".to_string()],
        &phrase_candidates,
        10,
        Bm25Params::default(),
    )
    .unwrap();
assert_eq!(hits[0].0, 2);
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
file-backed `postings::raw::RawSegmentFile` sealed from a live numeric postings
shard built with `RawTermDictionary`.

## File-backed BM25

Enable `raw-segment` to score numeric postings segments without opening a full
in-memory `InvertedIndex`:

```toml
[dependencies]
lexir = { version = "0.2", features = ["raw-segment"] }
```

Use this path for immutable segment sets or streaming ingestion where a bounded
live shard is searched with sealed raw files. The caller still owns term-id
mapping, commit publication, deletes, dictionary persistence, and segment merge
policy. See [docs/raw-bm25.md](docs/raw-bm25.md) for the detailed API map.

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

## License

MIT OR Apache-2.0
