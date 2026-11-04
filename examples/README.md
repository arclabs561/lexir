# lexir examples

These examples are small demonstrations on synthetic documents. Their output,
including the per-query times printed by `raw_bm25_benefits`, is not benchmark
evidence; use `cargo bench` for measurement.

## Where to start

| I want to... | Run |
|---|---|
| Run the README's in-memory BM25 example | `readme_examples` |
| Score BM25 over one file-backed raw segment with lexical terms | `raw_bm25_segment` |
| Stream documents into sealed raw files plus a live shard, then search both | `raw_bm25_generation` |
| See how many raw segments and blocks BM25 pruning skips | `raw_bm25_benefits` |

## Default features

| Example | What it shows |
|---|---|
| `readme_examples` | Builds a two-document `bm25::InvertedIndex`, retrieves one query term, and asserts the expected top document; mirrors the README usage block |

```sh
cargo run --release --example readme_examples
```

CI also runs `readme_examples` as an assertion check.

## Feature-gated examples

| Feature | Example | What it shows |
|---|---|---|
| `raw-segment` | `raw_bm25_segment` | Encodes three documents through `RawTermDictionary`, seals the live shard to one raw file, and prints the top BM25 hits for a two-term query |
| `raw-segment` | `raw_bm25_generation` | Adds documents to a `RawBm25LiveShard`, seals every three documents to a raw file, persists and reloads the dictionary sidecar, and prints BM25 hits across sealed files plus the live shard, with and without a candidate filter |
| `raw-segment` | `raw_bm25_benefits` | Prints a tab-separated table of median per-query time and segment/block seen, scored, and pruned counts for three deterministic fixtures, comparing pruned search against forced or unseeded search |

```sh
cargo run --release --features raw-segment --example raw_bm25_segment
cargo run --release --features raw-segment --example raw_bm25_generation
cargo run --release --features raw-segment --example raw_bm25_benefits
```

See [`docs/raw-bm25.md`](../docs/raw-bm25.md) for the raw BM25 design these
examples exercise.
