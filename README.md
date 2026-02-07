# lexir

Lexical IR (BM25/TF‑IDF) on top of postings lists.

- **Docs**: <https://docs.rs/lexir>
- **CI**: <https://github.com/arclabs561/lexir/actions>

## Feature Selection

- **`default`**: Includes `persistence`.
- **In-memory only**: disable default features, e.g. `lexir = { version = "0.1.0", default-features = false }`.

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
