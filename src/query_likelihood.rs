//! Query likelihood language model retrieval.
//!
//! Ranks documents by \(P(Q|D)\): the probability that a document language model generated the
//! query. This is a foundational probabilistic retrieval approach (Ponte & Croft, 1998).
//!
//! This implementation is **index-only** (no raw document text required): it uses the same
//! postings-backed corpus statistics as BM25/TF-IDF.

use crate::bm25::InvertedIndex;
use crate::query_terms::term_multiplicities;
use crate::ranking::top_k_finite_scored_docs;
use crate::Error;
use rankfns::{lm_smoothed_p, SmoothingMethod};
use std::collections::HashMap;

/// Query likelihood parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryLikelihoodParams {
    /// Smoothing method to use.
    pub smoothing: SmoothingMethod,
}

fn corpus_probability(
    term: &str,
    corpus_term_freqs: &HashMap<String, u32>,
    corpus_size: u32,
) -> f32 {
    if corpus_size == 0 {
        return 0.0;
    }
    let term_freq = corpus_term_freqs.get(term).copied().unwrap_or(0) as f32;
    term_freq / corpus_size as f32
}

fn score_jelinek_mercer(
    index: &InvertedIndex,
    doc_id: u32,
    query_terms: &[(&str, usize)],
    lambda: f32,
    corpus_term_freqs: &HashMap<String, u32>,
    corpus_size: u32,
) -> f32 {
    let mut log_score = 0.0;
    let doc_len = index.document_length(doc_id) as f32;

    for &(term, count) in query_terms {
        let tf = index.term_frequency(doc_id, term) as f32;
        let p_corpus = corpus_probability(term, corpus_term_freqs, corpus_size);
        let p_smoothed = lm_smoothed_p(
            tf,
            doc_len,
            p_corpus,
            SmoothingMethod::JelinekMercer { lambda },
        );
        if p_smoothed > 0.0 {
            let log_p = p_smoothed.ln();
            for _ in 0..count {
                log_score += log_p;
            }
        }
    }

    log_score
}

fn score_dirichlet(
    index: &InvertedIndex,
    doc_id: u32,
    query_terms: &[(&str, usize)],
    mu: f32,
    corpus_term_freqs: &HashMap<String, u32>,
    corpus_size: u32,
) -> f32 {
    let doc_length = index.document_length(doc_id) as f32;
    let mut log_score = 0.0;

    for &(term, count) in query_terms {
        let term_freq = index.term_frequency(doc_id, term) as f32;
        let p_corpus = corpus_probability(term, corpus_term_freqs, corpus_size);
        let p_smoothed = lm_smoothed_p(
            term_freq,
            doc_length,
            p_corpus,
            SmoothingMethod::Dirichlet { mu },
        );
        if p_smoothed > 0.0 {
            let log_p = p_smoothed.ln();
            for _ in 0..count {
                log_score += log_p;
            }
        }
    }

    log_score
}

/// Retrieve top-k documents for a tokenized query using query-likelihood language models.
pub fn retrieve_query_likelihood(
    index: &InvertedIndex,
    query_terms: &[String],
    k: usize,
    params: QueryLikelihoodParams,
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

    let (corpus_term_freqs, corpus_size) = index.corpus_stats_cached();
    let corpus_term_freqs = corpus_term_freqs.as_ref();
    let terms = term_multiplicities(query_terms);

    // Candidate docs: prefer postings-based candidates, but fall back to all docs
    // (smoothing can give non-zero mass even for non-matching docs).
    let mut candidates = index.candidates(query_terms);
    if candidates.is_empty() {
        candidates = index.document_ids().collect();
    }

    let results = candidates.into_iter().map(|doc_id| {
        let score = match params.smoothing {
            SmoothingMethod::JelinekMercer { lambda } => score_jelinek_mercer(
                index,
                doc_id,
                &terms,
                lambda,
                corpus_term_freqs,
                corpus_size,
            ),
            SmoothingMethod::Dirichlet { mu } => {
                score_dirichlet(index, doc_id, &terms, mu, corpus_term_freqs, corpus_size)
            }
        };
        (doc_id, score)
    });
    Ok(top_k_finite_scored_docs(results, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::InvertedIndex;

    #[test]
    fn query_likelihood_is_deterministic_on_ties() {
        let mut ix = InvertedIndex::new();
        // Two identical docs, query matches both equally.
        ix.add_document(1, &["a".into(), "b".into()]);
        ix.add_document(2, &["a".into(), "b".into()]);

        let hits = retrieve_query_likelihood(
            &ix,
            &["a".into()],
            10,
            QueryLikelihoodParams {
                smoothing: SmoothingMethod::default(),
            },
        )
        .unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[1].0, 2);
    }

    #[test]
    fn query_likelihood_duplicate_query_terms_preserve_scoring_weight() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into()]);
        ix.add_document(2, &["a".into(), "a".into()]);

        let single = retrieve_query_likelihood(
            &ix,
            &["a".into()],
            10,
            QueryLikelihoodParams {
                smoothing: SmoothingMethod::default(),
            },
        )
        .unwrap();
        let duplicated = retrieve_query_likelihood(
            &ix,
            &["a".into(), "a".into()],
            10,
            QueryLikelihoodParams {
                smoothing: SmoothingMethod::default(),
            },
        )
        .unwrap();

        assert_eq!(single.len(), duplicated.len());
        for ((single_doc, single_score), (dup_doc, dup_score)) in
            single.iter().zip(duplicated.iter())
        {
            assert_eq!(single_doc, dup_doc);
            assert!((dup_score - single_score * 2.0).abs() < 1e-6);
        }
    }
}
