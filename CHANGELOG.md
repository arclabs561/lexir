# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added a Criterion benchmark for BM25 retrieval over a Zipf-shaped synthetic
  corpus.

### Changed
- Bumped the optional `gramdex` dependency to 0.3.4.
- Bumped the `postings` dependency to 0.1.8.
- Sped up short BM25 retrieval queries by reusing average document length across
  candidate scoring. In the retrieval benchmark, `bm25_retrieve/terms/2` moved
  from `[2.3789 ms 2.3959 ms 2.4100 ms]` to
  `[2.2708 ms 2.2781 ms 2.2908 ms]`; `bm25_retrieve/terms/8` stayed within
  noise.

### Removed
- `query_pipeline` example and its `qexpr` / `qplan` dev-dependencies (both crates retired).

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
