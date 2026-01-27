//! `lexir`: lexical IR built on `postings`.
//!
//! This crate is meant to be the **shared** Tekne “lexical retrieval” building block:
//! - `postings` provides postings lists + candidate planning (no false negatives).
//! - `lexir` provides **scoring + ranking** (BM25 / TF-IDF) over `postings`.
//!
//! Scope:
//! - In-memory indexes
//! - Deterministic ranking (tie-break by doc id)
//! - Caller-provided token streams (so different products can choose tokenization)
//!
//! Non-goals:
//! - Storing document content (index-only)
//! - Phrase queries / positional postings
//! - Query language beyond “bag of terms”
//!
//! References:
//! - Robertson & Walker (1994): probabilistic retrieval foundations
//! - Robertson & Zaragoza (2009): BM25 and beyond
//! - Spärck Jones (1972): term specificity / IDF motivation

pub mod bm25;
#[cfg(feature = "fuzzy")]
pub mod fuzzy;
pub mod query_likelihood;
pub mod tfidf;

pub use error::Error;

mod error {
    /// Errors for lexical retrieval.
    #[derive(thiserror::Error, Debug)]
    pub enum Error {
        /// Query term list was empty.
        #[error("empty query")]
        EmptyQuery,
        /// Index contains no documents.
        #[error("empty index")]
        EmptyIndex,
        /// Fuzzy-expansion configuration was invalid.
        #[error("invalid fuzzy configuration: {0}")]
        InvalidFuzzyConfig(&'static str),
    }
}
