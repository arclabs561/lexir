//! Okapi BM25 over `postings::PostingsIndex`.
//!
//! This is the shared “lexical scorer” used by Tekne crates that want BM25:
//! - candidate generation comes from `postings` (with bailout support)
//! - scoring is standard BM25 with optional BM25L/BM25+ variants
//! - ranking is deterministic (score desc, then doc_id asc)
//!
//! References:
//! - Robertson & Walker (1994). "Some simple effective approximations to the 2-Poisson model..."
//! - Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond."

use crate::Error;
use postings::{CandidatePlan, PlannerConfig, PostingsIndex};
use rankfns::{bm25_idf_plus1, bm25_tf, Retriever};
use std::cell::RefCell;
use std::collections::HashMap;

impl Retriever for InvertedIndex {
    type Query = [String];
    type DocId = u32;

    fn retrieve(
        &self,
        query: &Self::Query,
        k: usize,
    ) -> Result<Vec<(Self::DocId, f32)>, Box<dyn std::error::Error>> {
        Ok(self.retrieve(query, k, Bm25Params::default())?)
    }
}

/// BM25 variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Bm25Variant {
    /// Standard BM25 (Okapi).
    #[default]
    Standard,
    /// BM25L: adds a small constant to TF contribution.
    BM25L {
        /// Additive term-frequency offset.
        delta: f32,
    },
    /// BM25+: lower-bounds TF contribution.
    BM25Plus {
        /// Additive term-frequency offset.
        delta: f32,
    },
}

impl Bm25Variant {
    /// Create BM25L with the conventional default delta (0.5).
    pub fn bm25l() -> Self {
        Self::BM25L { delta: 0.5 }
    }
    /// Create BM25L with a custom delta.
    pub fn bm25l_with_delta(delta: f32) -> Self {
        Self::BM25L { delta }
    }
    /// Create BM25+ with the conventional default delta (1.0).
    pub fn bm25plus() -> Self {
        Self::BM25Plus { delta: 1.0 }
    }
    /// Create BM25+ with a custom delta.
    pub fn bm25plus_with_delta(delta: f32) -> Self {
        Self::BM25Plus { delta }
    }
}

/// BM25 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation parameter.
    pub k1: f32,
    /// Length normalization parameter.
    pub b: f32,
    /// Variant choice (Standard/BM25L/BM25+).
    pub variant: Bm25Variant,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            variant: Bm25Variant::Standard,
        }
    }
}

impl Bm25Params {
    /// Create BM25L parameters with default delta (0.5).
    pub fn bm25l() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            variant: Bm25Variant::bm25l(),
        }
    }

    /// Create BM25+ parameters with default delta (1.0).
    pub fn bm25plus() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            variant: Bm25Variant::bm25plus(),
        }
    }
}

/// Inverted index for BM25 retrieval.
#[derive(Debug)]
pub struct InvertedIndex {
    postings: PostingsIndex<String>,
    // Lazily computed IDF cache (term -> idf), invalidated on write.
    precomputed_idf: RefCell<HashMap<String, f32>>,
    idf_computed_at_num_docs: RefCell<u32>,
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl InvertedIndex {
    /// Create a new empty BM25 index.
    pub fn new() -> Self {
        Self {
            postings: PostingsIndex::new(),
            precomputed_idf: RefCell::new(HashMap::new()),
            idf_computed_at_num_docs: RefCell::new(0),
        }
    }

    /// Create a BM25 index from an existing postings index.
    pub fn from_postings(postings: PostingsIndex<String>) -> Self {
        Self {
            postings,
            precomputed_idf: RefCell::new(HashMap::new()),
            idf_computed_at_num_docs: RefCell::new(0),
        }
    }

    /// Save the index using `durability` (crash-safe atomic write).
    #[cfg(feature = "persistence")]
    pub fn save<D: durability::Directory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        String: serde::Serialize,
    {
        self.postings.save(dir, path)
    }

    /// Save the index with stable-storage durability barriers.
    ///
    /// For filesystem-backed directories, this fsyncs the temp file and syncs the
    /// parent directory after the atomic rename. For non-filesystem backends this
    /// returns `NotSupported`.
    #[cfg(feature = "persistence")]
    pub fn save_durable<D: durability::DurableDirectory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        String: serde::Serialize,
    {
        self.postings.save_durable(dir, path)
    }

    /// Load an index using `durability`.
    #[cfg(feature = "persistence")]
    pub fn load<D: durability::Directory + ?Sized>(
        dir: &D,
        path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        for<'de> String: serde::Deserialize<'de>,
    {
        let postings = PostingsIndex::<String>::load(dir, path)?;
        Ok(Self::from_postings(postings))
    }

    /// Count of live documents currently indexed.
    pub fn num_docs(&self) -> u32 {
        self.postings.num_docs()
    }

    /// Iterate live document ids.
    pub fn document_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.postings.document_ids()
    }

    /// Add/update a document by doc id and token stream.
    pub fn add_document(&mut self, doc_id: u32, terms: &[String]) {
        // Model updates as delete+add (segment-style).
        let _ = self.postings.delete_document(doc_id);
        let _ = self.postings.add_document(doc_id, terms);
        self.precomputed_idf.borrow_mut().clear();
        *self.idf_computed_at_num_docs.borrow_mut() = 0;
    }

    /// Delete a document by id.
    ///
    /// Returns whether the document existed.
    pub fn delete_document(&mut self, doc_id: u32) -> bool {
        let deleted = self.postings.delete_document(doc_id);
        if deleted {
            self.precomputed_idf.borrow_mut().clear();
            *self.idf_computed_at_num_docs.borrow_mut() = 0;
        }
        deleted
    }

    /// Term frequency of `term` in `doc_id` (0 if doc missing / term absent).
    pub fn term_frequency(&self, doc_id: u32, term: &str) -> u32 {
        self.postings.term_frequency(doc_id, term)
    }

    /// Document frequency of `term` over live docs.
    pub fn doc_frequency(&self, term: &str) -> u32 {
        self.postings.df(term)
    }

    /// Document length (in terms). Returns 0 for unknown doc ids.
    pub fn document_length(&self, doc_id: u32) -> u32 {
        self.postings.document_len(doc_id)
    }

    /// Iterate all distinct terms seen in live documents.
    pub fn terms(&self) -> impl Iterator<Item = &str> + '_ {
        self.postings.terms().map(|t| t.as_str())
    }

    /// Iterate postings (doc_id, tf) for a term across all segments (live docs only).
    pub fn postings_iter<'a>(&'a self, term: &'a str) -> impl Iterator<Item = (u32, u32)> + 'a {
        self.postings.postings_iter(term)
    }

    /// Average document length (in terms) over live docs.
    pub fn avg_doc_len(&self) -> f32 {
        self.postings.avg_doc_len()
    }

    /// Candidate documents: docs that contain at least one query term, with bailout.
    pub fn candidates(&self, query_terms: &[String]) -> Vec<u32> {
        match self
            .postings
            .plan_candidates(query_terms, PlannerConfig::default())
        {
            CandidatePlan::Candidates(c) => c,
            CandidatePlan::ScanAll => {
                let mut v: Vec<u32> = self.document_ids().collect();
                v.sort_unstable();
                v
            }
        }
    }

    fn ensure_idf_computed(&self) {
        let computed_at = *self.idf_computed_at_num_docs.borrow();
        if computed_at == self.num_docs() {
            let idf_map = self.precomputed_idf.borrow();
            if !idf_map.is_empty() {
                return;
            }
        }

        let mut idf_map = self.precomputed_idf.borrow_mut();
        idf_map.clear();
        let n = self.num_docs() as f32;
        for term in self.postings.terms() {
            let df_f = self.postings.df(term) as f32;
            if df_f > 0.0 {
                let idf = bm25_idf_plus1(n, df_f);
                idf_map.insert(term.to_string(), idf);
            }
        }
        *self.idf_computed_at_num_docs.borrow_mut() = self.num_docs();
    }

    /// IDF with BM25 “+1” variant (positive idf, stable for frequent terms).
    pub fn idf(&self, term: &str) -> f32 {
        {
            let idf_map = self.precomputed_idf.borrow();
            if let Some(&idf) = idf_map.get(term) {
                return idf;
            }
        }
        let df = self.postings.df(term) as f32;
        let n = self.num_docs() as f32;
        bm25_idf_plus1(n, df)
    }

    /// BM25 score for a document (caller provides tokenized query terms).
    pub fn score(&self, doc_id: u32, query_terms: &[String], params: Bm25Params) -> f32 {
        let avg_doc_len = self.postings.avg_doc_len();
        if avg_doc_len == 0.0 {
            return 0.0;
        }

        let doc_length = self.postings.document_len(doc_id) as f32;
        let mut score = 0.0;

        for term in query_terms {
            let idf = self.idf(term);
            if idf == 0.0 {
                continue;
            }
            let tf = self.postings.term_frequency(doc_id, term) as f32;
            if tf == 0.0 {
                continue;
            }

            let mut tf_score = bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b);

            match params.variant {
                Bm25Variant::Standard => {}
                Bm25Variant::BM25L { delta } => tf_score += delta,
                Bm25Variant::BM25Plus { delta } => tf_score += delta,
            }

            score += idf * tf_score;
        }
        score
    }

    /// Retrieve top-k documents using BM25 scoring.
    ///
    /// - **Input**: caller-provided tokenized query terms.
    /// - **Candidates**: generated via `postings` with a bailout (may scan all docs for very broad queries).
    /// - **Output**: sorted deterministically by `(score desc, doc_id asc)`.
    pub fn retrieve(
        &self,
        query_terms: &[String],
        k: usize,
        params: Bm25Params,
    ) -> Result<Vec<(u32, f32)>, Error> {
        if query_terms.is_empty() {
            return Err(Error::EmptyQuery);
        }
        if self.num_docs() == 0 {
            return Err(Error::EmptyIndex);
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        self.ensure_idf_computed();
        let query_idfs: Vec<f32> = query_terms.iter().map(|t| self.idf(t)).collect();
        let candidates = self.candidates(query_terms);

        // Min-heap top-k.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct FloatOrd(f32);
        impl Eq for FloatOrd {}
        impl PartialOrd for FloatOrd {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for FloatOrd {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<Reverse<(FloatOrd, u32)>> = BinaryHeap::with_capacity(k + 1);
        for doc_id in candidates {
            let score = score_optimized(self, doc_id, query_terms, &query_idfs, params);
            if !score.is_finite() || score <= 0.0 {
                continue;
            }
            if heap.len() < k {
                heap.push(Reverse((FloatOrd(score), doc_id)));
            } else if let Some(&Reverse((FloatOrd(min_score), _))) = heap.peek() {
                if score > min_score {
                    heap.pop();
                    heap.push(Reverse((FloatOrd(score), doc_id)));
                }
            }
        }

        let mut results: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|Reverse((FloatOrd(score), doc_id))| (doc_id, score))
            .collect();

        // Deterministic: score desc, then doc_id asc.
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(results)
    }
}

fn score_optimized(
    index: &InvertedIndex,
    doc_id: u32,
    query_terms: &[String],
    query_idfs: &[f32],
    params: Bm25Params,
) -> f32 {
    let avg_doc_len = index.postings.avg_doc_len();
    if avg_doc_len == 0.0 {
        return 0.0;
    }
    let doc_length = index.postings.document_len(doc_id) as f32;
    let mut score = 0.0;
    for (term, &idf) in query_terms.iter().zip(query_idfs.iter()) {
        if idf == 0.0 {
            continue;
        }
        let tf = index.postings.term_frequency(doc_id, term) as f32;
        if tf == 0.0 {
            continue;
        }
        let mut tf_score = bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b);
        match params.variant {
            Bm25Variant::Standard => {}
            Bm25Variant::BM25L { delta } => tf_score += delta,
            Bm25Variant::BM25Plus { delta } => tf_score += delta,
        }
        score += idf * tf_score;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieve_tie_breaks_by_doc_id() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into(), "x".into()]);
        ix.add_document(2, &["a".into(), "x".into()]);

        let hits = ix
            .retrieve(&["a".into()], 10, Bm25Params::default())
            .unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[1].0, 2);
    }

    #[test]
    fn candidates_scan_all_is_sorted() {
        let mut ix = InvertedIndex::new();
        // Make "common" broad enough to trigger the default bailout.
        for doc_id in 0..10u32 {
            ix.add_document(doc_id, &["common".into(), format!("u{doc_id}")]);
        }
        let cands = ix.candidates(&["common".into()]);
        let mut expected: Vec<u32> = (0..10u32).collect();
        expected.sort_unstable();
        assert_eq!(cands, expected);
    }
}
