# lexir

[![crates.io](https://img.shields.io/crates/v/lexir.svg)](https://crates.io/crates/lexir)
[![Documentation](https://docs.rs/lexir/badge.svg)](https://docs.rs/lexir)
[![CI](https://github.com/arclabs561/lexir/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/lexir/actions/workflows/ci.yml)

Lexical scoring over postings lists.

**Status**: experimental. Published on `crates.io`; the API may still shift.

## Feature Selection

- **`default`**: Includes `persistence`.
- **In-memory only**: disable default features.

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

`lexir::raw::retrieve_bm25_raw_file` scores one segment using that segment's
local document frequencies. For an immutable segment set, use
`retrieve_bm25_raw_files` or build `RawBm25CorpusStats` once and pass it to
`retrieve_bm25_raw_files_with_stats`, so every segment uses the same IDF and
average document length. The multi-file path orders segments by a conservative
BM25 upper bound and can skip segments that cannot enter the current top-k.
Use `retrieve_bm25_raw_file_candidates` or
`retrieve_bm25_raw_files_candidates` when an exact filter has already produced
candidate doc ids, for example from `postings::positional::raw` phrase or NEAR
segment-set helpers. Use the positional `_filtered` helpers when the exact
filter also needs tombstones or newer-version masks. Use
`retrieve_bm25_raw_files_candidates_and_index` for exact candidates over sealed
raw files plus one live raw postings shard; use
`retrieve_bm25_raw_files_filtered_candidates_and_index` when sealed files can
contain stale copies hidden by a lifecycle layer.
Use `retrieve_bm25_raw_file_filtered` or
`retrieve_bm25_raw_files_filtered` when a lifecycle layer exposes tombstones or
newer-version masks. The predicate is applied while building BM25 corpus stats
and while scoring, so stale docs cannot fill a top-k slot before filtering.
For streaming ingestion, `retrieve_bm25_raw_files_and_index` searches sealed
raw segment files plus one live `postings::PostingsIndex<u64, u32>` shard with
shared BM25 corpus stats, scoring the live shard first so its top-k threshold
can skip low-bound sealed files.
Use `retrieve_bm25_raw_files_filtered_and_index` when sealed files need a
visibility predicate but the live shard is already current.
Use `retrieve_bm25_raw_file_with_search_stats` for single-segment traversal
diagnostics and `retrieve_bm25_raw_files_with_search_stats` for searched/pruned
segment counts. `retrieve_bm25_raw_files_with_diagnostics` also aggregates
file traversal counters such as raw posting blocks seen, scored, and pruned;
`cargo run --release --example raw_bm25_benefits --features raw-segment` prints
those counters for deterministic pruning fixtures. `RawBm25CorpusStats::from_raw_files_all_terms`
builds reusable stats for all terms in a raw segment generation without reading
postings payloads; `from_raw_files_and_index_all_terms` does the same for sealed
files plus one live shard. File-backed BM25 can also use raw posting-block
metadata to skip blocks that cannot enter the current top-k. Segment document
ids must already be globally unique.

`RawTermDictionary` is an in-process adapter from lexical terms to numeric raw
term ids. `insert` assigns ids in insertion order; `from_terms_sorted` assigns
ids from sorted unique terms for reproducible offline builds. Export
`RawTermDictionary::terms` as a term-id-ordered sidecar and reload it with
`RawTermDictionary::from_terms_in_id_order`.
`RawBm25LiveShard` pairs that dictionary with one bounded live
`postings::PostingsIndex<u64, u32>` shard, can add token streams, encode
queries, decide when the live shard reaches a caller-selected
`RawBm25SealPolicy`, and seal the live shard to a caller-provided raw segment
writer. It does not own paths, commits, durable publication, retention, or
compaction.

`cargo run --example raw_bm25_generation --features raw-segment` shows the
streaming ingestion pattern: encode lexical documents into a live numeric shard,
seal full shards to raw files, persist the dictionary sidecar, reload it, and
search sealed files plus the remaining live shard, including a prefiltered
candidate-ranking pass.

The caller owns commit lifecycle, deletes, dictionary persistence, and segment
merge policy.

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
