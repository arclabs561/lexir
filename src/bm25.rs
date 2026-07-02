//! Okapi BM25 over `postings::PostingsIndex`.
//!
//! Shared lexical scorer for crates that want BM25:
//! - candidate generation comes from `postings` (with bailout support)
//! - scoring is standard BM25 with optional BM25L/BM25+ variants
//! - ranking is deterministic (score desc, then doc_id asc)
//!
//! References:
//! - Robertson & Walker (1994). "Some simple effective approximations to the 2-Poisson model..."
//! - Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond."

use crate::query_terms::term_multiplicities;
use crate::ranking::top_k_positive_scored_docs;
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
    pub fn save<D: durability::storage::Directory + ?Sized>(
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
    pub fn save_durable<D: durability::storage::Directory + ?Sized>(
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
    pub fn load<D: durability::storage::Directory + ?Sized>(
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

    fn ensure_idf_cache_current(&self) {
        let n = self.num_docs();
        let mut computed_at = self.idf_computed_at_num_docs.borrow_mut();
        if *computed_at != n {
            self.precomputed_idf.borrow_mut().clear();
            *computed_at = n;
        }
    }

    /// IDF with BM25 “+1” variant (positive idf, stable for frequent terms).
    pub fn idf(&self, term: &str) -> f32 {
        self.ensure_idf_cache_current();
        {
            let idf_map = self.precomputed_idf.borrow();
            if let Some(&idf) = idf_map.get(term) {
                return idf;
            }
        }
        let df = self.postings.df(term);
        let n = self.num_docs();
        let idf = bm25_idf_plus1(n, df);
        if df > 0 {
            self.precomputed_idf
                .borrow_mut()
                .insert(term.to_string(), idf);
        }
        idf
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

            let tf_score = bm25_tf_score(tf, doc_length, avg_doc_len, params);

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

        let terms = term_multiplicities(query_terms);
        let mut touched_upper_bound = 0usize;
        let query_idfs: Vec<f32> = terms
            .iter()
            .map(|&(term, _)| {
                let idf = self.idf(term);
                if idf != 0.0 {
                    touched_upper_bound =
                        touched_upper_bound.saturating_add(self.postings.df(term) as usize);
                }
                idf
            })
            .collect();
        let avg_doc_len = self.postings.avg_doc_len();
        if avg_doc_len == 0.0 {
            return Ok(Vec::new());
        }

        let mut scores = HashMap::with_capacity(touched_upper_bound.min(self.num_docs() as usize));
        for (&(term, count), &idf) in terms.iter().zip(query_idfs.iter()) {
            if idf == 0.0 {
                continue;
            }

            for (doc_id, tf) in self.postings.postings_iter(term) {
                let doc_length = self.postings.document_len(doc_id) as f32;
                let tf_score = bm25_tf_score(tf as f32, doc_length, avg_doc_len, params);
                let contribution = idf * tf_score;
                if contribution != 0.0 {
                    let score = scores.entry(doc_id).or_insert(0.0);
                    for _ in 0..count {
                        *score += contribution;
                    }
                }
            }
        }

        Ok(top_k_positive_scored_docs(scores, k))
    }
}

fn bm25_tf_score(tf: f32, doc_length: f32, avg_doc_len: f32, params: Bm25Params) -> f32 {
    match params.variant {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    #[test]
    fn idf_cache_is_populated_lazily_by_query_terms() {
        let mut ix = InvertedIndex::new();
        ix.add_document(0, &["alpha".into(), "beta".into()]);
        ix.add_document(1, &["beta".into(), "gamma".into()]);

        assert!(
            ix.precomputed_idf.borrow().is_empty(),
            "writes leave the lazy IDF cache empty"
        );

        let _ = ix
            .retrieve(&["alpha".into()], 10, Bm25Params::default())
            .unwrap();
        assert_eq!(
            ix.precomputed_idf.borrow().len(),
            1,
            "first query caches only the queried term, not the full vocabulary"
        );
        assert!(ix.precomputed_idf.borrow().contains_key("alpha"));

        let _ = ix
            .retrieve(&["beta".into()], 10, Bm25Params::default())
            .unwrap();
        assert_eq!(
            ix.precomputed_idf.borrow().len(),
            2,
            "later queries extend the cache per term"
        );

        ix.add_document(2, &["delta".into()]);
        assert!(
            ix.precomputed_idf.borrow().is_empty(),
            "writes invalidate the query-term IDF cache"
        );
    }

    #[test]
    fn retrieve_duplicate_query_terms_preserve_scoring_weight() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into()]);
        ix.add_document(2, &["a".into(), "a".into()]);

        let single = ix
            .retrieve(&["a".into()], 10, Bm25Params::default())
            .unwrap();
        let duplicated = ix
            .retrieve(&["a".into(), "a".into()], 10, Bm25Params::default())
            .unwrap();

        assert_eq!(single.len(), duplicated.len());
        for ((single_doc, single_score), (dup_doc, dup_score)) in
            single.iter().zip(duplicated.iter())
        {
            assert_eq!(single_doc, dup_doc);
            assert!((dup_score - single_score * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn retrieve_interleaved_duplicate_terms_match_point_scores() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["a".into(), "b".into()]);
        ix.add_document(2, &["a".into(), "a".into(), "b".into()]);
        let query = vec!["a".into(), "b".into(), "a".into()];

        let hits = ix.retrieve(&query, 10, Bm25Params::default()).unwrap();

        for (doc_id, score) in hits {
            let expected = ix.score(doc_id, &query, Bm25Params::default());
            assert!(
                (score - expected).abs() < 1e-6,
                "doc {doc_id}: retrieve={score}, point_score={expected}"
            );
        }
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

    // -- Property-based tests --

    fn arb_term() -> impl Strategy<Value = String> {
        "[a-z]{1,6}".prop_map(|s| s)
    }

    fn arb_doc_terms(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(arb_term(), min..max)
    }

    proptest! {
        #[test]
        fn retrieve_matches_exhaustive_score_sort(
            docs in prop::collection::vec(arb_doc_terms(0, 10), 1..12),
            query_terms in prop::collection::vec(arb_term(), 1..5),
            k in 0usize..20,
        ) {
            let mut ix = InvertedIndex::new();
            for (doc_id, terms) in docs.iter().enumerate() {
                ix.add_document(doc_id as u32, terms);
            }

            let mut expected: Vec<(u32, f32)> = ix
                .document_ids()
                .filter_map(|doc_id| {
                    let score = ix.score(doc_id, &query_terms, Bm25Params::default());
                    (score.is_finite() && score > 0.0).then_some((doc_id, score))
                })
                .collect();
            expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            expected.truncate(k);

            let got = ix.retrieve(&query_terms, k, Bm25Params::default()).unwrap();

            prop_assert_eq!(got, expected);
        }

        #[test]
        fn score_is_non_negative(
            doc_terms in arb_doc_terms(1, 10),
            query_terms in prop::collection::vec(arb_term(), 1..4),
        ) {
            let mut ix = InvertedIndex::new();
            ix.add_document(0, &doc_terms);
            let score = ix.score(0, &query_terms, Bm25Params::default());
            prop_assert!(score >= 0.0, "BM25 score must be non-negative, got {}", score);
        }

        #[test]
        fn add_document_increments_num_docs(
            docs in prop::collection::vec(arb_doc_terms(1, 8), 1..10),
        ) {
            let mut ix = InvertedIndex::new();
            for (i, terms) in docs.iter().enumerate() {
                ix.add_document(i as u32, terms);
                prop_assert_eq!(ix.num_docs(), (i + 1) as u32);
            }
        }

        #[test]
        fn retrieve_returns_at_most_num_docs(
            doc_count in 1..8usize,
            k in 1..20usize,
        ) {
            let mut ix = InvertedIndex::new();
            for i in 0..doc_count {
                ix.add_document(i as u32, &["shared".into(), format!("unique{i}")]);
            }
            let results = ix.retrieve(&["shared".into()], k, Bm25Params::default()).unwrap();
            prop_assert!(results.len() <= doc_count, "got {} results for {} docs", results.len(), doc_count);
        }

        #[test]
        fn results_sorted_by_score_desc(
            doc_count in 2..8usize,
        ) {
            let mut ix = InvertedIndex::new();
            for i in 0..doc_count {
                // Varying term frequency so scores differ.
                let mut terms: Vec<String> = vec!["common".into()];
                for _ in 0..i {
                    terms.push("common".into());
                }
                terms.push(format!("filler{i}"));
                ix.add_document(i as u32, &terms);
            }
            let results = ix.retrieve(&["common".into()], 20, Bm25Params::default()).unwrap();
            for w in results.windows(2) {
                prop_assert!(
                    w[0].1 >= w[1].1,
                    "results not sorted: {:?} before {:?}", w[0], w[1],
                );
            }
        }

        #[test]
        fn superset_terms_score_higher(
            extra_terms in prop::collection::vec(arb_term(), 1..4),
        ) {
            // Two docs with the same length: one has all query terms, the other has a subset.
            let query: Vec<String> = vec!["alpha".into(), "beta".into()];
            let mut full_doc: Vec<String> = query.clone();
            let mut partial_doc: Vec<String> = vec!["alpha".into()];

            // Pad both to the same length with filler terms to control for length normalization.
            let target_len = full_doc.len() + extra_terms.len();
            for t in &extra_terms {
                full_doc.push(t.clone());
            }
            while partial_doc.len() < target_len {
                partial_doc.push("zzzfiller".into());
            }

            let mut ix = InvertedIndex::new();
            ix.add_document(0, &full_doc);
            ix.add_document(1, &partial_doc);

            let score_full = ix.score(0, &query, Bm25Params::default());
            let score_partial = ix.score(1, &query, Bm25Params::default());
            prop_assert!(
                score_full >= score_partial,
                "full={} should be >= partial={}", score_full, score_partial,
            );
        }
    }
}
