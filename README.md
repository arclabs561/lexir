# lexir

Lexical scoring over postings lists

**Status:** Experimental. The API may change.

`lexir` ranks caller-provided token streams. Its in-memory index owns postings
and corpus statistics; the `raw-segment` feature scores immutable
`postings::raw` files. Tokenization, term IDs, positional matching, document
storage, commits, deletes, and segment merges belong to the caller or
[`postings`](https://crates.io/crates/postings).

The default BM25 comes from [`rankfns`](https://crates.io/crates/rankfns):
Lucene's IDF with the `(k1 + 1)` numerator and exact document lengths; BM25L
and BM25+ are opt-in variants. For a full search engine
with its own tokenizers, query parser and segment lifecycle, use tantivy; use
`lexir` when you supply the tokens and want the scoring layer alone.

## Install

```toml
[dependencies]
lexir = "0.4"
```

## Usage

```rust
use lexir::bm25::InvertedIndex;

let mut index = InvertedIndex::new();
index.add_document(1, &["red".into(), "fox".into()]);
index.add_document(2, &["blue".into(), "whale".into()]);

let hits = index.retrieve(&["fox".into()], 10, Default::default())?;
assert_eq!(hits[0].0, 1);
# Ok::<(), lexir::Error>(())
```

## Features

- `persistence` (default): save and load in-memory indexes
- `raw-segment`: score file-backed raw segments; see the [guide](docs/raw-bm25.md)
- `recordlog`: append operations for rebuilding an index
- `cli`: build and search indexes from the command line
- `fuzzy`: expand out-of-vocabulary query terms

Run `cargo run --features cli -- --help` for CLI usage.

## License

MIT OR Apache-2.0
