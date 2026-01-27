//! TF-IDF scoring over the shared postings-backed index.
//!
//! This module intentionally reuses the same postings-backed document statistics as BM25,
//! so callers can compare BM25 vs TF-IDF with identical tokenization and corpus stats.
//!
//! References:
//! - Spärck Jones (1972): term specificity / IDF motivation.

use crate::bm25::InvertedIndex;
use crate::Error;
use rankfns as rf;

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

/// Term-frequency transform variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TfVariant {
    /// Linear TF: `tf = f_{t,d}`.
    Linear,
    /// Log-scaled TF: `tf = 1 + ln(f_{t,d})` for `f_{t,d} > 0`.
    LogScaled,
}

/// IDF transform variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdfVariant {
    /// Standard IDF: `ln(N / df)`.
    Standard,
    /// Smoothed IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))` (BM25-style, stable).
    Smoothed,
}

fn compute_tf(tf_count: u32, variant: TfVariant) -> f32 {
    let v = match variant {
        TfVariant::Linear => rf::TfVariant::Linear,
        TfVariant::LogScaled => rf::TfVariant::LogScaled,
    };
    rf::tf_transform(tf_count, v)
}

fn compute_idf(num_docs: u32, doc_frequency: u32, variant: IdfVariant) -> f32 {
    let v = match variant {
        IdfVariant::Standard => rf::IdfVariant::Standard,
        IdfVariant::Smoothed => rf::IdfVariant::Smoothed,
    };
    rf::idf_transform(num_docs, doc_frequency, v)
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
        let tf_count = index.term_frequency(doc_id, term) as u32;
        if tf_count == 0 {
            continue;
        }
        let tf = compute_tf(tf_count, params.tf_variant);
        let df = index.doc_frequency(term);
        let idf = compute_idf(num_docs, df, params.idf_variant);
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

    let candidates = index.candidates(query_terms);
    let mut scored: Vec<(u32, f32)> = candidates
        .into_iter()
        .map(|doc_id| (doc_id, score_tfidf(index, doc_id, query_terms, params)))
        .filter(|(_, score)| score.is_finite() && *score > 0.0)
        .collect();

    // Deterministic: score desc, then doc_id asc.
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if scored.len() > k {
        scored.truncate(k);
    }
    Ok(scored)
}
