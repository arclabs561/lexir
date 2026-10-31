# lexir

Lexical scoring over postings lists

**Status:** Experimental. The API may change.

`lexir` ranks token streams that you supply. Its in-memory index stores postings
and corpus statistics. The `raw-segment` feature scores immutable
[`postings`](https://crates.io/crates/postings) files. You supply tokenization
and document storage; positional matching and segment lifecycle are separate.

The default BM25 comes from [`rankfns`](https://crates.io/crates/rankfns):
Lucene's IDF with the `(k1 + 1)` numerator and exact document lengths; BM25L
and BM25+ are opt-in variants. The variants apply the term-frequency transforms
of Lv and Zhai (2011) but keep Lucene's IDF, `ln((N + 1) / (df + 0.5))`, where
the paper uses `ln((N + 1) / df)`, so their rankings can differ from the paper.

## Install

```toml
[dependencies]
lexir = "0.5"
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

The default and non-CLI library features support Rust 1.80. The `cli` feature
and all-feature builds require Rust 1.85 with the current Clap dependency.

Run `cargo run --features cli -- --help` for CLI usage.

## License

MIT OR Apache-2.0
