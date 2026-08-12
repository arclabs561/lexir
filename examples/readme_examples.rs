//! Compile-check of the README usage example.
//! Run: cargo run --example readme_examples

use lexir::bm25::InvertedIndex;

fn main() -> Result<(), lexir::Error> {
    let mut index = InvertedIndex::new();
    index.add_document(1, &["red".into(), "fox".into()]);
    index.add_document(2, &["blue".into(), "whale".into()]);

    let hits = index.retrieve(&["fox".into()], 10, Default::default())?;
    assert_eq!(hits[0].0, 1);
    Ok(())
}
