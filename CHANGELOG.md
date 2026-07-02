# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
