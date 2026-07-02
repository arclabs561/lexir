//! TF-IDF scoring over the shared postings-backed index.
//!
//! This module intentionally reuses the same postings-backed document statistics as BM25,
//! so callers can compare BM25 vs TF-IDF with identical tokenization and corpus stats.
//!
//! References:
//! - Spärck Jones (1972): term specificity / IDF motivation.

use crate::bm25::InvertedIndex;
use crate::query_terms::term_multiplicities;
use crate::ranking::top_k_positive_scored_docs;
use crate::Error;
use rankfns::{idf_transform, tf_transform, IdfVariant, TfVariant};
use std::collections::HashMap;

/// TF-IDF parameters.
#[derive(Debug, Clone, Copy)]
pub struct TfIdfParams {
    /// Term-frequency transform.
    pub tf_variant: TfVariant,
    /// IDF transform.
    pub idf_variant: IdfVariant,
}

impl Default for TfIdfParams {
    fn default() -> Self {
        Self {
            tf_variant: TfVariant::LogScaled,
            idf_variant: IdfVariant::Standard,
        }
    }
}

impl TfIdfParams {
    /// Create TF-IDF parameters with linear TF and standard IDF.
    pub fn linear() -> Self {
        Self {
            tf_variant: TfVariant::Linear,
            idf_variant: IdfVariant::Standard,
        }
    }

    /// Create TF-IDF parameters with log-scaled TF and smoothed IDF.
    pub fn smoothed() -> Self {
        Self {
            tf_variant: TfVariant::LogScaled,
            idf_variant: IdfVariant::Smoothed,
        }
    }
}

/// TF-IDF score for a document, given tokenized query terms.
pub fn score_tfidf(
    index: &InvertedIndex,
    doc_id: u32,
    query_terms: &[String],
    params: TfIdfParams,
) -> f32 {
    let mut score = 0.0;
    let num_docs = index.num_docs();
    for term in query_terms {
        let tf_count = index.term_frequency(doc_id, term);
        if tf_count == 0 {
            continue;
        }
        let tf = tf_transform(tf_count, params.tf_variant);
        let df = index.doc_frequency(term);
        let idf = idf_transform(num_docs, df, params.idf_variant);
        if idf == 0.0 {
            continue;
        }
        score += tf * idf;
    }
    score
}

/// Retrieve top-k documents using TF-IDF.
pub fn retrieve_tfidf(
    index: &InvertedIndex,
    query_terms: &[String],
    k: usize,
    params: TfIdfParams,
) -> Result<Vec<(u32, f32)>, Error> {
    if query_terms.is_empty() {
        return Err(Error::EmptyQuery);
    }
    if index.num_docs() == 0 {
        return Err(Error::EmptyIndex);
    }
    if k == 0 {
        return Ok(Vec::new());
    }

    let num_docs = index.num_docs();
    let mut touched_upper_bound = 0usize;
    let terms = term_multiplicities(query_terms);
    let query_idfs: Vec<f32> = terms
        .iter()
        .map(|&(term, _)| {
            let df = index.doc_frequency(term);
            let idf = idf_transform(num_docs, df, params.idf_variant);
            if idf != 0.0 {
                touched_upper_bound = touched_upper_bound.saturating_add(df as usize);
            }
            idf
        })
        .collect();

    let mut scores = HashMap::with_capacity(touched_upper_bound.min(num_docs as usize));
    for (&(term, count), &idf) in terms.iter().zip(query_idfs.iter()) {
        if idf == 0.0 {
            continue;
        }

        for (doc_id, tf_count) in index.postings_iter(term) {
            let tf = tf_transform(tf_count, params.tf_variant);
            let contribution = tf * idf;
            if contribution != 0.0 {
                let score = scores.entry(doc_id).or_insert(0.0);
                *score += contribution * count as f32;
            }
        }
    }

    Ok(top_k_positive_scored_docs(scores, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::InvertedIndex;

    #[test]
    fn tfidf_tie_breaks_by_doc_id() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into()]);
        ix.add_document(2, &["a".into()]);
        // Ensure df(a) < N so IDF is non-zero under standard IDF.
        ix.add_document(3, &["x".into()]);

        let hits = retrieve_tfidf(&ix, &["a".into()], 10, TfIdfParams::linear()).unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[1].0, 2);
    }

    #[test]
    fn tfidf_duplicate_query_terms_preserve_scoring_weight() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into()]);
        ix.add_document(2, &["a".into(), "a".into()]);
        // Ensure df(a) < N so IDF is non-zero under standard IDF.
        ix.add_document(3, &["x".into()]);

        let single = retrieve_tfidf(&ix, &["a".into()], 10, TfIdfParams::linear()).unwrap();
        let duplicated =
            retrieve_tfidf(&ix, &["a".into(), "a".into()], 10, TfIdfParams::linear()).unwrap();

        assert_eq!(single.len(), duplicated.len());
        for ((single_doc, single_score), (dup_doc, dup_score)) in
            single.iter().zip(duplicated.iter())
        {
            assert_eq!(single_doc, dup_doc);
            assert!((dup_score - single_score * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn tfidf_interleaved_duplicate_terms_match_point_scores() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into(), "b".into()]);
        ix.add_document(2, &["a".into(), "a".into(), "b".into()]);
        ix.add_document(3, &["x".into()]);
        let query = vec!["a".into(), "b".into(), "a".into()];

        let hits = retrieve_tfidf(&ix, &query, 10, TfIdfParams::linear()).unwrap();

        for (doc_id, score) in hits {
            let expected = score_tfidf(&ix, doc_id, &query, TfIdfParams::linear());
            assert!(
                (score - expected).abs() < 1e-6,
                "doc {doc_id}: retrieve={score}, point_score={expected}"
            );
        }
    }
}
