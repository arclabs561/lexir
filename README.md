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

## Performance

BM25 retrieval on the in-memory `InvertedIndex`, release build, `k = 10`,
`Bm25Params::default()`. The benchmark corpus is fixed at 20,000 documents of 80
Zipf-distributed terms drawn from a 5,000-term vocabulary (1.6M postings); query
terms are taken in term-id order and only terms with `df >= 20` are used. Because
the benchmark fixes the corpus and `k`, the number of query terms is the
retrieval parameter that moves in these rows.

| Operation | Corpus / query terms | Time (mean) |
| --- | --- | --- |
| `bm25_retrieve/terms/1` | 20,000 docs / 1 term | 604 µs |
| `bm25_retrieve/terms/2` | 20,000 docs / 2 terms | 1.24 ms |
| `bm25_retrieve/terms/8` | 20,000 docs / 8 terms | 3.68 ms |
| `bm25_retrieve/terms/32` | 20,000 docs / 32 terms | 8.54 ms |

Machine: Apple M3 Max, 16 cores, 128 GiB RAM, macOS 27.0.1, aarch64-apple-darwin,
rustc 1.99.1, Criterion 0.5.

```console
cargo bench --bench retrieve -- 'bm25_retrieve/terms/(1|2|8|32)$' \
  --warm-up-time 2 --measurement-time 10 --sample-size 50
```

Not claimed: no latency guarantee, no cold-start or single-query figures, no
corpus sizes other than 20,000 documents, and no comparison against other search
engines. The host was running unrelated work during measurement, so treat these
as one run on a shared machine; a repeat of the 2-term and 8-term rows gave
1.18 ms and 3.65 ms.

## License

MIT OR Apache-2.0
