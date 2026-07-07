# File-backed BM25

Enable `raw-segment` to score numeric `postings::raw` segment files without
opening a full in-memory `InvertedIndex`:

```toml
[dependencies]
lexir = { version = "0.2", features = ["raw-segment"] }
```

## Segment Sets

`lexir::raw::retrieve_bm25_raw_file` scores one segment using that segment's
local document frequencies. For an immutable segment set, use
`retrieve_bm25_raw_files` or build `RawBm25CorpusStats` once and pass it to
`retrieve_bm25_raw_files_with_stats`, so every segment uses the same IDF and
average document length.

The multi-file path orders segments by a conservative BM25 upper bound and can
skip segments that cannot enter the current top-k. Segment document ids must be
globally unique.

## Prefiltered Candidates

Use `retrieve_bm25_raw_file_candidates` or
`retrieve_bm25_raw_files_candidates` when an exact filter has already produced
candidate doc ids, for example from `postings::positional::raw` phrase or NEAR
segment-set helpers.

Use the positional `_filtered` helpers when the exact filter also needs
tombstones or newer-version masks. The predicate is applied while building BM25
corpus stats and while scoring, so stale docs cannot fill a top-k slot before
filtering.

## Streaming Ingestion

For streaming ingestion, `retrieve_bm25_raw_files_and_index` searches sealed raw
segment files plus one live `postings::PostingsIndex<u64, u32>` shard with
shared BM25 corpus stats. The live shard is scored first so its top-k threshold
can skip low-bound sealed files.

`RawTermDictionary` adapts lexical terms to numeric raw term ids. Export
`RawTermDictionary::terms` as a term-id-ordered sidecar and reload it with
`RawTermDictionary::from_terms_in_id_order`.

`RawBm25LiveShard` pairs that dictionary with one bounded live shard, can add
token streams, encode queries, decide when the shard reaches a caller-selected
`RawBm25SealPolicy`, and seal the shard to a caller-provided raw segment writer.
It does not own paths, commits, durable publication, retention, or compaction.

`cargo run --example raw_bm25_generation --features raw-segment` shows the
pattern: encode documents into a live numeric shard, seal full shards to raw
files, persist the dictionary sidecar, reload it, and search sealed files plus
the remaining live shard.

## Diagnostics

Use `retrieve_bm25_raw_file_with_search_stats` for single-segment traversal
diagnostics and `retrieve_bm25_raw_files_with_search_stats` for searched/pruned
segment counts. `retrieve_bm25_raw_files_with_diagnostics` also aggregates file
traversal counters such as raw posting blocks seen, scored, and pruned.

`cargo run --release --example raw_bm25_benefits --features raw-segment` prints
those counters for deterministic pruning fixtures.
