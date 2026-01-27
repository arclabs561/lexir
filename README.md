# lexir

Lexical IR (BM25/TF‑IDF) on top of postings lists.

- **Docs**: <https://docs.rs/lexir>
- **CI**: <https://github.com/arclabs561/lexir/actions>

## What it is

`lexir` is the scoring/ranking layer. Candidate generation and storage live in `postings`.

## Usage (library)

```rust
use lexir::bm25::InvertedIndex;

let mut idx = InvertedIndex::new();
idx.add_document(1, &["hello".to_string(), "world".to_string()]);
let hits = idx.retrieve(&["hello".to_string()], 10, Default::default()).unwrap();
assert_eq!(hits[0].0, 1);
```

## Features

- `persistence` (default): save/load via `durability` + `postings/persistence`
- `recordlog`: append-only operation logs for rebuildable indexes (CLI uses this)
- `cli`: enables the `lexir` CLI (debugging + end-to-end validation)
- `fuzzy`: fuzzy query expansion via `gramdex`

