//! Query likelihood language model retrieval.
//!
//! Ranks documents by \(P(Q|D)\): the probability that a document language model generated the
//! query. This is a foundational probabilistic retrieval approach (Ponte & Croft, 1998).
//!
//! This implementation is **index-only** (no raw document text required): it uses the same
//! postings-backed corpus statistics as BM25/TF-IDF.

use crate::bm25::InvertedIndex;
use crate::Error;
use rankfns as rf;
use std::collections::{HashMap, HashSet};

/// Smoothing method for query likelihood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// Jelinek-Mercer smoothing: interpolates document and corpus language models.
    ///
    /// `lambda` is clamped to `[0, 1]`.
    JelinekMercer {
        /// Interpolation weight.
        lambda: f32,
    },
    /// Dirichlet smoothing: Bayesian approach with automatic length adaptation.
    ///
    /// `mu` is clamped to `>= 0`.
    Dirichlet {
        /// Prior strength.
        mu: f32,
    },
}

impl Default for SmoothingMethod {
    fn default() -> Self {
        Self::Dirichlet { mu: 1000.0 }
    }
}

impl SmoothingMethod {
    /// Create Jelinek-Mercer smoothing with default lambda (0.5).
    pub fn jelinek_mercer() -> Self {
        Self::JelinekMercer { lambda: 0.5 }
    }

    /// Create Jelinek-Mercer smoothing with custom lambda.
    pub fn jelinek_mercer_with_lambda(lambda: f32) -> Self {
        Self::JelinekMercer {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Create Dirichlet smoothing with default mu (1000.0).
    pub fn dirichlet() -> Self {
        Self::Dirichlet { mu: 1000.0 }
    }

    /// Create Dirichlet smoothing with custom mu.
    pub fn dirichlet_with_mu(mu: f32) -> Self {
        Self::Dirichlet { mu: mu.max(0.0) }
    }
}

/// Query likelihood parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryLikelihoodParams {
    /// Smoothing method to use.
    pub smoothing: SmoothingMethod,
}

fn compute_corpus_stats(index: &InvertedIndex) -> (HashMap<String, u32>, u32) {
    let mut corpus_term_freqs: HashMap<String, u32> = HashMap::new();
    let mut corpus_size: u32 = 0;

    for term in index.terms() {
        let total_tf: u32 = index.postings_iter(term).map(|(_doc, tf)| tf).sum();
        corpus_term_freqs.insert(term.to_string(), total_tf);
        corpus_size += total_tf;
    }

    (corpus_term_freqs, corpus_size)
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
    query_terms: &[String],
    lambda: f32,
    corpus_term_freqs: &HashMap<String, u32>,
    corpus_size: u32,
) -> f32 {
    let mut log_score = 0.0;

    for term in query_terms {
        let doc_len = index.document_length(doc_id) as f32;
        let tf = index.term_frequency(doc_id, term) as f32;
        let p_corpus = corpus_probability(term, corpus_term_freqs, corpus_size);
        let p_smoothed = rf::lm_smoothed_p(
            tf,
            doc_len,
            p_corpus,
            rf::SmoothingMethod::JelinekMercer { lambda },
        );
        if p_smoothed > 0.0 {
            log_score += p_smoothed.ln();
        }
    }

    log_score
}

fn score_dirichlet(
    index: &InvertedIndex,
    doc_id: u32,
    query_terms: &[String],
    mu: f32,
    corpus_term_freqs: &HashMap<String, u32>,
    corpus_size: u32,
) -> f32 {
    let doc_length = index.document_length(doc_id) as f32;
    let mut log_score = 0.0;

    for term in query_terms {
        let term_freq = index.term_frequency(doc_id, term) as f32;
        let p_corpus = corpus_probability(term, corpus_term_freqs, corpus_size);
        let p_smoothed = rf::lm_smoothed_p(
            term_freq,
            doc_length,
            p_corpus,
            rf::SmoothingMethod::Dirichlet { mu },
        );
        if p_smoothed > 0.0 {
            log_score += p_smoothed.ln();
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

    let (corpus_term_freqs, corpus_size) = compute_corpus_stats(index);

    // Candidate docs: prefer postings-based candidates, but fall back to all docs
    // (smoothing can give non-zero mass even for non-matching docs).
    let mut candidates: HashSet<u32> = index.candidates(query_terms).into_iter().collect();
    if candidates.is_empty() {
        candidates = index.document_ids().collect();
    }

    let mut results: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
    for doc_id in candidates {
        let score = match params.smoothing {
            SmoothingMethod::JelinekMercer { lambda } => score_jelinek_mercer(
                index,
                doc_id,
                query_terms,
                lambda,
                &corpus_term_freqs,
                corpus_size,
            ),
            SmoothingMethod::Dirichlet { mu } => score_dirichlet(
                index,
                doc_id,
                query_terms,
                mu,
                &corpus_term_freqs,
                corpus_size,
            ),
        };

        if score.is_finite() {
            results.push((doc_id, score));
        }
    }

    // Deterministic: score desc, then doc_id asc.
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if results.len() > k {
        results.truncate(k);
    }
    Ok(results)
}
