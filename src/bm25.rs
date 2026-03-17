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
use rankfns::{bm25_idf_plus1, bm25_tf};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct CachedCorpusStats {
    term_freqs: Arc<HashMap<String, u32>>,
    corpus_size: u32,
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
    // Lazily computed corpus term frequencies for query-likelihood retrieval.
    corpus_stats: RefCell<Option<CachedCorpusStats>>,
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
            corpus_stats: RefCell::new(None),
        }
    }

    /// Create a BM25 index from an existing postings index.
    pub fn from_postings(postings: PostingsIndex<String>) -> Self {
        Self {
            postings,
            precomputed_idf: RefCell::new(HashMap::new()),
            idf_computed_at_num_docs: RefCell::new(0),
            corpus_stats: RefCell::new(None),
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
        *self.corpus_stats.borrow_mut() = None;
    }

    /// Delete a document by id.
    ///
    /// Returns whether the document existed.
    pub fn delete_document(&mut self, doc_id: u32) -> bool {
        let deleted = self.postings.delete_document(doc_id);
        if deleted {
            self.precomputed_idf.borrow_mut().clear();
            *self.idf_computed_at_num_docs.borrow_mut() = 0;
            *self.corpus_stats.borrow_mut() = None;
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

    pub(crate) fn corpus_stats_cached(&self) -> (Arc<HashMap<String, u32>>, u32) {
        if let Some(cs) = self.corpus_stats.borrow().as_ref() {
            return (cs.term_freqs.clone(), cs.corpus_size);
        }

        let mut corpus_term_freqs: HashMap<String, u32> = HashMap::new();
        let mut corpus_size: u32 = 0;

        for term in self.terms() {
            let total_tf: u32 = self.postings_iter(term).map(|(_doc, tf)| tf).sum();
            corpus_term_freqs.insert(term.to_string(), total_tf);
            corpus_size = corpus_size.saturating_add(total_tf);
        }

        let term_freqs = Arc::new(corpus_term_freqs);
        *self.corpus_stats.borrow_mut() = Some(CachedCorpusStats {
            term_freqs: term_freqs.clone(),
            corpus_size,
        });
        (term_freqs, corpus_size)
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
        let n = self.num_docs();
        for term in self.postings.terms() {
            let df = self.postings.df(term);
            if df > 0 {
                let idf = bm25_idf_plus1(n, df);
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
        let df = self.postings.df(term);
        let n = self.num_docs();
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

            let tf_score = match params.variant {
                Bm25Variant::Standard => bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b),
                Bm25Variant::BM25L { delta } => {
                    // BM25L (Lv & Zhai, 2011): length-normalize TF first,
                    // add delta, then apply saturation.
                    let b = params.b.clamp(0.0, 1.0);
                    let ctf = tf / (1.0 - b + b * doc_length / avg_doc_len.max(1e-9)).max(1e-9);
                    let tf_l = ctf + delta;
                    (tf_l * (params.k1 + 1.0)) / (tf_l + params.k1).max(1e-9)
                }
                Bm25Variant::BM25Plus { delta } => {
                    // BM25+ (Lv & Zhai, 2011): standard BM25 TF, then add delta.
                    bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b) + delta
                }
            };

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
        let tf_score = match params.variant {
            Bm25Variant::Standard => bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b),
            Bm25Variant::BM25L { delta } => {
                let b = params.b.clamp(0.0, 1.0);
                let ctf = tf / (1.0 - b + b * doc_length / avg_doc_len.max(1e-9)).max(1e-9);
                let tf_l = ctf + delta;
                (tf_l * (params.k1 + 1.0)) / (tf_l + params.k1).max(1e-9)
            }
            Bm25Variant::BM25Plus { delta } => {
                bm25_tf(tf, doc_length, avg_doc_len, params.k1, params.b) + delta
            }
        };
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

    fn build_test_index() -> InvertedIndex {
        let mut ix = InvertedIndex::new();
        // doc 0: short doc, high TF for "neural"
        ix.add_document(0, &["neural".into(), "neural".into(), "network".into()]);
        // doc 1: long doc, same terms spread out
        ix.add_document(
            1,
            &[
                "neural".into(),
                "network".into(),
                "deep".into(),
                "learning".into(),
                "optimization".into(),
                "gradient".into(),
            ],
        );
        // doc 2: short, different topic
        ix.add_document(2, &["cat".into(), "dog".into()]);
        ix
    }

    #[test]
    fn bm25l_differs_from_standard() {
        let ix = build_test_index();
        let query = vec!["neural".into()];

        let standard = ix.retrieve(&query, 3, Bm25Params::default()).unwrap();
        let bm25l = ix.retrieve(&query, 3, Bm25Params::bm25l()).unwrap();

        // Both should return doc 0 and doc 1 (both contain "neural")
        assert_eq!(standard.len(), 2);
        assert_eq!(bm25l.len(), 2);

        // BM25L scores should differ from standard
        let std_score_0 = standard.iter().find(|(id, _)| *id == 0).unwrap().1;
        let bm25l_score_0 = bm25l.iter().find(|(id, _)| *id == 0).unwrap().1;
        assert!(
            (std_score_0 - bm25l_score_0).abs() > 1e-6,
            "BM25L should produce different scores than Standard: std={}, bm25l={}",
            std_score_0,
            bm25l_score_0
        );
    }

    #[test]
    fn bm25l_differs_from_bm25plus() {
        let ix = build_test_index();
        let query = vec!["neural".into()];

        let bm25l = ix.retrieve(&query, 3, Bm25Params::bm25l()).unwrap();
        let bm25plus = ix.retrieve(&query, 3, Bm25Params::bm25plus()).unwrap();

        // Both should return results
        assert!(!bm25l.is_empty());
        assert!(!bm25plus.is_empty());

        // BM25L and BM25+ should produce DIFFERENT scores (they are distinct algorithms)
        let l_score = bm25l[0].1;
        let plus_score = bm25plus[0].1;
        assert!(
            (l_score - plus_score).abs() > 1e-6,
            "BM25L and BM25+ should differ: l={}, plus={}",
            l_score,
            plus_score
        );
    }

    #[test]
    fn bm25plus_adds_delta_to_standard() {
        let ix = build_test_index();
        let query = vec!["neural".into()];
        let delta = 1.0;

        let standard = ix.retrieve(&query, 3, Bm25Params::default()).unwrap();
        let bm25plus = ix
            .retrieve(
                &query,
                3,
                Bm25Params {
                    variant: Bm25Variant::bm25plus_with_delta(delta),
                    ..Bm25Params::default()
                },
            )
            .unwrap();

        // BM25+ score should be higher than standard (adds delta)
        let std_score = standard[0].1;
        let plus_score = bm25plus[0].1;
        assert!(
            plus_score > std_score,
            "BM25+ should score higher: std={}, plus={}",
            std_score,
            plus_score
        );
    }

    #[test]
    fn bm25l_reduces_length_penalty() {
        // BM25L should reduce the length penalty for long documents
        let ix = build_test_index();
        let query = vec!["neural".into()];

        let standard = ix.retrieve(&query, 3, Bm25Params::default()).unwrap();
        let bm25l = ix.retrieve(&query, 3, Bm25Params::bm25l()).unwrap();

        // doc 1 is longer than doc 0. BM25L should penalize length less,
        // so the score ratio (doc1/doc0) should be higher for BM25L.
        let std_0 = standard.iter().find(|(id, _)| *id == 0).unwrap().1;
        let std_1 = standard.iter().find(|(id, _)| *id == 1).unwrap().1;
        let l_0 = bm25l.iter().find(|(id, _)| *id == 0).unwrap().1;
        let l_1 = bm25l.iter().find(|(id, _)| *id == 1).unwrap().1;

        let std_ratio = std_1 / std_0;
        let l_ratio = l_1 / l_0;

        // BM25L should give long docs (doc 1) a relatively higher score
        assert!(
            l_ratio > std_ratio,
            "BM25L should reduce length penalty: std_ratio={}, l_ratio={}",
            std_ratio,
            l_ratio
        );
    }
}
