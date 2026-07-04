//! Compile-check of README usage examples.
//! Run: cargo run --example readme_examples

use lexir::bm25::{Bm25Params, InvertedIndex};
use lexir::query_likelihood::{retrieve_query_likelihood, QueryLikelihoodParams};
use lexir::tfidf::{retrieve_tfidf, TfIdfParams};

fn main() {
    let mut idx = InvertedIndex::new();
    idx.add_document(1, &["hello".to_string(), "world".to_string()]);
    idx.add_document(2, &["hello".to_string(), "foo".to_string()]);
    idx.add_document(3, &["other".to_string()]); // ensures IDF(hello) > 0 for TF-IDF

    // BM25
    let hits = idx
        .retrieve(&["hello".to_string()], 10, Default::default())
        .unwrap();
    assert!(!hits.is_empty() && hits[0].0 == 1, "BM25 example");

    // BM25 over prefiltered candidates
    let hits = idx
        .retrieve_candidates(&["hello".to_string()], &[2], 10, Bm25Params::default())
        .unwrap();
    assert_eq!(hits[0].0, 2, "prefiltered BM25 example");

    // TF-IDF
    let hits = retrieve_tfidf(&idx, &["hello".to_string()], 10, TfIdfParams::linear()).unwrap();
    assert!(!hits.is_empty() && hits[0].0 == 1, "TF-IDF example");

    // Query Likelihood
    let hits = retrieve_query_likelihood(
        &idx,
        &["hello".to_string()],
        10,
        QueryLikelihoodParams::default(),
    )
    .unwrap();
    assert!(!hits.is_empty(), "Query Likelihood example");

    println!("All README examples compile and run correctly.");
}
