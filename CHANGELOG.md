# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `RawTermDictionary` for mapping lexical terms to `postings::raw` term
  ids and encoding raw BM25 documents or queries.

### Changed

- Changed file-backed raw BM25 retrieval to use raw posting-block metadata for
  top-k block pruning on broad nonnegative BM25 queries.

## [0.2.5] - 2026-07-03

### Changed

- Changed all-term raw BM25 stats construction to stream raw segment term
  metadata from postings instead of allocating a full term-id vector or doing a
  second directory lookup per term.

## [0.2.4] - 2026-07-03

### Added

- Added `RawBm25CorpusStats::from_raw_files_all_terms` for reusable BM25 stats
  over immutable raw segment sets without reading postings payloads.
- Added raw BM25 benchmarks for all-term stats construction and repeated
  multi-file searches with precomputed all-term stats.
- Added a raw BM25 benchmark for many small segment files with precomputed
  corpus stats.

### Changed

- Changed raw BM25 dense score and document-length buffers to use the segment's
  document-id span instead of allocating from zero to the maximum document id.
- Changed raw BM25 range planning to use postings' raw segment `doc_id_range`
  API instead of deriving the same span by scanning document lengths.

## [0.2.3] - 2026-07-03

### Changed

- Changed raw BM25 examples and benchmarks to use postings' sorted raw-document
  writer for increasing-document-id streams, avoiding postings' whole-corpus
  document map during raw segment setup.

## [0.2.2] - 2026-07-03

### Added
- Added raw BM25 parity sweeps across query lengths, retrieval depths, BM25
  variants, sparse document ids, and multi-file global statistics.
- Added a raw BM25 pruning regression that checks a dominated segment is
  skipped while preserving the in-memory ranking.
- Added `retrieve_bm25_raw_files_with_search_stats` for raw BM25 segment-pruning
  diagnostics.

### Changed
- Changed raw BM25 path selection to ignore absent raw terms before choosing
  between streaming postings and cached document-length scoring.
- Changed file-backed raw BM25 broad-query scoring to prefill dense
  document-length caches from raw segment metadata instead of doing repeated
  per-document metadata lookups.
- Changed multi-file raw BM25 retrieval to skip finite zero-bound segments before
  scoring, avoiding useless posting reads for absent query terms.
- Changed raw BM25 multi-file benchmarks to keep segment-reference construction
  outside timed loops.
- Changed raw BM25 examples and benchmarks to write postings raw segments
  directly to files through postings' caller-sink writer APIs.

## [0.2.1] - 2026-07-03

### Added
- Added the `raw-segment` feature with `retrieve_bm25_raw_segment` and
  `retrieve_bm25_raw_file`, scoring BM25 directly over postings' immutable raw
  segments (in-memory or file-backed) without building an inverted index.
- Added corpus-level raw BM25 stats and multi-file raw-segment retrieval helpers
  so immutable file-backed segments can be scored as one corpus.
- Added a `raw_bm25_segment` example showing BM25 scoring over a file-backed
  `postings::raw::RawSegmentFile`.
- Added raw-segment BM25 cases to the retrieval Criterion benchmark.

### Changed
- Changed raw-segment BM25 scoring to use dense score and document-length
  accumulators for dense doc-id segments, while keeping the sparse HashMap path
  for high-id sparse segments.
- Changed raw-segment BM25 scoring to stream postings with document lengths for
  short raw queries while keeping the document-length cache path for
  expansion-heavy queries.

## [0.2.0] - 2026-07-02

### Changed
- Changed `lexir index` and `lexir search` to stream corpus files line by
  line instead of reading the whole input file before indexing.
- Sped up duplicate-heavy retrieval queries by folding query-term
  multiplicities before traversing postings. Duplicate query terms still carry
  the same scoring weight, but BM25, TF-IDF, and query-likelihood scan each
  unique posting list once. In the focused duplicate-query benchmark,
  `bm25_retrieve/duplicate_terms/8` moved from
  `[4.1767 ms 4.1823 ms 4.1878 ms]` to
  `[1.1405 ms 1.1437 ms 1.1456 ms]`,
  `tfidf_retrieve/duplicate_terms/8` moved from
  `[2.7521 ms 2.7666 ms 2.7819 ms]` to
  `[756.67 us 758.24 us 759.87 us]`, and
  `query_likelihood_retrieve/duplicate_terms/8` moved from
  `[10.357 ms 10.427 ms 10.515 ms]` to
  `[2.8002 ms 2.8048 ms 2.8118 ms]`.
- Reduced BM25's duplicate-query accumulation overhead by applying folded query
  multiplicity with one multiply per posting. In the focused duplicate-query
  benchmark, `bm25_retrieve/duplicate_terms/8` moved from
  `[1.1405 ms 1.1437 ms 1.1456 ms]` to
  `[1.1009 ms 1.1027 ms 1.1047 ms]`.
- Sped up TF-IDF retrieval by accumulating scores directly from postings lists
  instead of materializing candidates and then doing per-document term-frequency
  lookups. Current timings on the retrieval benchmark are
  `tfidf_retrieve/terms/2` at `[768.07 us 769.63 us 771.25 us]` and
  `tfidf_retrieve/terms/8` at `[2.4533 ms 2.4594 ms 2.4657 ms]`.
- Sped up query-likelihood retrieval by keeping postings-planner candidates as a
  vector and using bounded finite-score top-k ranking instead of hashing the
  candidates and sorting the full scored set. Current timings on the retrieval
  benchmark are `query_likelihood_retrieve/terms/2` at
  `[3.0050 ms 3.0253 ms 3.0461 ms]` and
  `query_likelihood_retrieve/terms/8` at `[12.443 ms 12.597 ms 12.754 ms]`.

## [0.1.5] - 2026-07-02

### Added
- Added a Criterion benchmark for BM25 retrieval over a Zipf-shaped synthetic
  corpus.

### Changed
- Bumped the optional `gramdex` dependency to 0.3.4.
- Bumped the `postings` dependency to 0.1.8.
- Changed BM25 IDF caching to populate lazily per queried term instead of
  precomputing the whole vocabulary on the first retrieval.
- Bumped the optional `durability` dependency to 0.6.12 and reused its
  `Directory::delete_durable` helper for durable materialized-log cleanup.
- Sped up short BM25 retrieval queries by reusing average document length across
  candidate scoring. In the retrieval benchmark, `bm25_retrieve/terms/2` moved
  from `[2.3789 ms 2.3959 ms 2.4100 ms]` to
  `[2.2708 ms 2.2781 ms 2.2908 ms]`; `bm25_retrieve/terms/8` stayed within
  noise.
- Sped up BM25 retrieval by accumulating scores directly from postings lists
  instead of building candidates and then doing per-document term-frequency
  lookups. In the retrieval benchmark, `bm25_retrieve/terms/2` moved from
  `[2.2708 ms 2.2781 ms 2.2908 ms]` to
  `[1.1353 ms 1.1386 ms 1.1425 ms]`, and `bm25_retrieve/terms/8` moved from
  `[8.4996 ms 8.5904 ms 8.6477 ms]` to
  `[3.6293 ms 3.6378 ms 3.6485 ms]`.

### Removed
- `query_pipeline` example and its `qexpr` / `qplan` dev-dependencies (both crates retired).

### Fixed
- Synced the parent directory after deleting an empty materialized log when
  `--durable` is requested, matching the durability strength of log rewrites
  that publish by atomic rename.

## [0.1.4] - 2026-06-11

### Changed
- Bumped the `postings` dependency to 0.1.6, `durability` to 0.6, and `rankfns` to 0.1.3.
- Removed internal ecosystem references from doc comments.
- Fleshed out CONTRIBUTING.md (setup, style, testing, PR expectations).

## [0.1.3] - 2026-04-06

### Added
- `qexpr` + `qplan` query-pipeline example.
- Property-based tests for BM25 retrieval.
- `[workspace]` table for standalone builds.

### Changed
- Condensed the README to prose and trimmed feature lists.
- Upgraded the `durability` dependency to 0.5.0.
- Cached corpus stats for query-likelihood scoring.
- Used `rankfns` kernels and removed duplicated variants.

### Fixed
- BM25L implementation (was identical to BM25+).
- Stabilized tie ordering.
