//! BM25 retrieval over `postings::raw` numeric segments.
//!
//! This is the out-of-core lexical path: callers keep their own term-id mapping
//! and use `postings` raw segments for storage, while `lexir` supplies BM25
//! scoring and deterministic top-k ranking. `RawTermDictionary` is the optional
//! in-process adapter from lexical terms to raw numeric term ids; it does not
//! own persistence, commits, deletes, or segment merges.

use crate::bm25::{bm25_tf_score, Bm25Params, Bm25Variant};
use crate::ranking::top_k_positive_scored_docs;
use postings::raw::{
    RawPostingBlockMeta, RawSegment, RawSegmentFile, RawSegmentFileError, RawTermId,
};
use postings::{DocId, PostingsIndex};
use rankfns::bm25_idf_plus1;
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Errors returned when encoding lexical terms into raw term ids.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RawTermDictionaryError {
    /// A single document contains more occurrences of a term than fit in the
    /// raw segment frequency type.
    #[error("raw term frequency overflow for term id {term_id}")]
    TermFrequencyOverflow { term_id: RawTermId },
    /// A persisted raw term-id dictionary listed the same lexical term twice.
    #[error("duplicate raw dictionary term at term id {term_id}")]
    DuplicateTerm { term_id: RawTermId },
}

/// In-process mapping from lexical terms to `postings::raw` term ids.
///
/// Raw postings segments store numeric term ids. This helper gives callers a
/// focused lexicon boundary for examples, tests, and build pipelines that do
/// not already have their own term dictionary. Ids assigned by `insert` follow
/// insertion order; use `from_terms_sorted` when corpus-order-independent ids
/// matter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RawTermDictionary {
    terms: Vec<Arc<str>>,
    ids: HashMap<Arc<str>, RawTermId>,
}

impl RawTermDictionary {
    /// Create an empty term dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a dictionary whose ids follow sorted unique term order.
    ///
    /// This is the reproducible-build constructor: the same set of terms gets
    /// the same ids regardless of corpus traversal order.
    pub fn from_terms_sorted<I, T>(terms: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut terms: Vec<String> = terms
            .into_iter()
            .map(|term| term.as_ref().to_owned())
            .collect();
        terms.sort_unstable();
        terms.dedup();

        let mut dictionary = Self::new();
        for term in terms {
            dictionary.insert(term);
        }
        dictionary
    }

    /// Create a dictionary from terms already ordered by raw term id.
    ///
    /// Use this when loading a persisted dictionary sidecar written from
    /// [`RawTermDictionary::terms`]. Duplicate lexical terms are rejected so a
    /// corrupt sidecar cannot silently shift term ids.
    pub fn from_terms_in_id_order<I, T>(terms: I) -> Result<Self, RawTermDictionaryError>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut dictionary = Self::new();
        for term in terms {
            let term = term.as_ref();
            if dictionary.ids.contains_key(term) {
                return Err(RawTermDictionaryError::DuplicateTerm {
                    term_id: dictionary.terms.len() as RawTermId,
                });
            }
            dictionary.insert(term);
        }
        Ok(dictionary)
    }

    /// Number of terms in the dictionary.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Return true when the dictionary has no terms.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Insert a term if absent and return its raw term id.
    ///
    /// New ids follow insertion order.
    pub fn insert(&mut self, term: impl AsRef<str>) -> RawTermId {
        let term = term.as_ref();
        if let Some(&term_id) = self.ids.get(term) {
            return term_id;
        }

        let term_id = self.terms.len() as RawTermId;
        let term: Arc<str> = Arc::from(term);
        self.terms.push(Arc::clone(&term));
        self.ids.insert(term, term_id);
        term_id
    }

    /// Return the raw term id for a lexical term.
    pub fn id(&self, term: &str) -> Option<RawTermId> {
        self.ids.get(term).copied()
    }

    /// Return the lexical term for a raw term id.
    pub fn term(&self, term_id: RawTermId) -> Option<&str> {
        usize::try_from(term_id)
            .ok()
            .and_then(|index| self.terms.get(index))
            .map(AsRef::as_ref)
    }

    /// Iterate terms in raw term-id order.
    pub fn terms(&self) -> impl Iterator<Item = (RawTermId, &str)> + '_ {
        self.terms
            .iter()
            .enumerate()
            .map(|(term_id, term)| (term_id as RawTermId, term.as_ref()))
    }

    /// Encode a document as sorted `(raw_term_id, term_frequency)` pairs.
    ///
    /// Unknown terms are inserted. Duplicate terms are counted. The returned
    /// vector is sorted by raw term id, which matches the raw segment writer's
    /// preferred document shape.
    pub fn encode_document<I, T>(
        &mut self,
        terms: I,
    ) -> Result<Vec<(RawTermId, u32)>, RawTermDictionaryError>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut counts = BTreeMap::new();
        for term in terms {
            let term_id = self.insert(term);
            let count = counts.entry(term_id).or_insert(0u32);
            *count = count
                .checked_add(1)
                .ok_or(RawTermDictionaryError::TermFrequencyOverflow { term_id })?;
        }
        Ok(counts.into_iter().collect())
    }

    /// Encode a query as raw term ids.
    ///
    /// Unknown terms are omitted. Duplicate known terms are preserved so BM25
    /// scoring keeps the caller's query-term multiplicity.
    pub fn encode_query<I, T>(&self, terms: I) -> Vec<RawTermId>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        terms
            .into_iter()
            .filter_map(|term| self.id(term.as_ref()))
            .collect()
    }
}

/// Errors returned by raw-segment lexical scoring.
#[derive(Debug)]
pub enum RawScoringError<E> {
    /// Query term list was empty.
    EmptyQuery,
    /// Segment contains no documents.
    EmptyIndex,
    /// A scored term appears in a segment but has no corpus document-frequency
    /// entry in the provided stats.
    MissingCorpusStats(RawTermId),
    /// The underlying raw segment reader failed.
    Source(E),
}

impl<E: fmt::Display> fmt::Display for RawScoringError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => f.write_str("empty query"),
            Self::EmptyIndex => f.write_str("empty index"),
            Self::MissingCorpusStats(term_id) => {
                write!(f, "missing corpus stats for raw term {term_id}")
            }
            Self::Source(source) => source.fmt(f),
        }
    }
}

impl<E> std::error::Error for RawScoringError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(source) => Some(source),
            Self::EmptyQuery | Self::EmptyIndex | Self::MissingCorpusStats(_) => None,
        }
    }
}

/// Query-scoped corpus statistics for raw-segment BM25 scoring.
///
/// BM25 IDF and length normalization are corpus-level quantities. For a single
/// raw segment, `retrieve_bm25_raw_file` derives these from that segment. For a
/// segment set, build stats from all searched segments and pass them to the
/// `*_with_stats` functions so every segment uses the same IDF and average
/// document length.
#[derive(Debug, Clone)]
pub struct RawBm25CorpusStats {
    num_docs: u32,
    avg_doc_len: f32,
    dfs: HashMap<RawTermId, u32>,
}

/// Segment-level diagnostics for a multi-file raw BM25 search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawBm25SearchStats {
    /// Raw segment files supplied to the search.
    pub segments_seen: usize,
    /// Raw segment files actually scored.
    pub segments_scored: usize,
    /// Raw segment files skipped by a zero bound or current top-k threshold.
    pub segments_pruned: usize,
}

/// Search diagnostics for a multi-file raw BM25 search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawBm25SearchDiagnostics {
    /// Segment pruning diagnostics for the searched raw segment files.
    pub segments: RawBm25SearchStats,
    /// Sum of unique query terms scored across raw files that were searched.
    pub terms_scored: usize,
    /// Sum of per-file posting upper bounds for scored terms.
    pub touched_postings_upper_bound: usize,
    /// Sum of dense accumulator slots allocated by dense file paths.
    pub dense_slots: usize,
    /// Sum of per-term posting blocks considered by block-pruned file paths.
    pub term_blocks_seen: usize,
    /// Sum of per-term posting blocks decoded by block-pruned file paths.
    pub term_blocks_scored: usize,
    /// Sum of per-term posting blocks skipped by block-pruned file paths.
    pub term_blocks_pruned: usize,
}

/// Hits and segment-level diagnostics from a multi-file raw BM25 search.
#[derive(Clone, Debug, PartialEq)]
pub struct RawBm25SearchResult {
    /// Top-k hits sorted by descending BM25 score, then document id.
    pub hits: Vec<(DocId, f32)>,
    /// Segment-pruning diagnostics for the search.
    pub stats: RawBm25SearchStats,
}

/// Hits and search diagnostics from a multi-file raw BM25 search.
#[derive(Clone, Debug, PartialEq)]
pub struct RawBm25DiagnosticSearchResult {
    /// Top-k hits sorted by descending BM25 score, then document id.
    pub hits: Vec<(DocId, f32)>,
    /// Segment and file traversal diagnostics for the search.
    pub diagnostics: RawBm25SearchDiagnostics,
}

impl RawBm25SearchDiagnostics {
    fn add_file_stats(&mut self, stats: RawBm25FileSearchStats) {
        self.terms_scored += stats.terms_scored;
        self.touched_postings_upper_bound += stats.touched_postings_upper_bound;
        self.dense_slots += stats.dense_slots;
        self.term_blocks_seen += stats.term_blocks_seen;
        self.term_blocks_scored += stats.term_blocks_scored;
        self.term_blocks_pruned += stats.term_blocks_pruned;
    }
}

/// File-backed raw BM25 traversal path used for a single segment search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum RawBm25FileSearchPath {
    /// No scoring path was needed because the request or matching terms were empty.
    #[default]
    Noop,
    /// Dense accumulator path using streamed `(doc_id, tf, doc_len)` postings.
    DenseStream,
    /// Dense accumulator path using postings plus cached document lengths.
    DenseCachedLengths,
    /// Sparse accumulator path using streamed `(doc_id, tf, doc_len)` postings.
    SparseStream,
    /// Sparse accumulator path using postings plus cached document lengths.
    SparseCachedLengths,
    /// File-backed block-pruned path using dense accumulators.
    BlockPrunedDense,
    /// File-backed block-pruned path using sparse accumulators.
    BlockPrunedSparse,
}

/// Diagnostics for a single file-backed raw BM25 segment search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct RawBm25FileSearchStats {
    /// Traversal path selected for the segment.
    pub path: RawBm25FileSearchPath,
    /// Unique query terms that were present and scored after IDF filtering.
    pub terms_scored: usize,
    /// Sum of per-term document frequencies for scored terms.
    pub touched_postings_upper_bound: usize,
    /// Dense accumulator slots allocated by dense paths.
    pub dense_slots: usize,
    /// Per-term posting blocks considered by block-pruned paths.
    pub term_blocks_seen: usize,
    /// Per-term posting blocks decoded by block-pruned paths.
    pub term_blocks_scored: usize,
    /// Per-term posting blocks skipped by block-pruned paths.
    pub term_blocks_pruned: usize,
}

/// Hits and traversal diagnostics from a single file-backed raw BM25 search.
#[derive(Clone, Debug, PartialEq)]
pub struct RawBm25FileSearchResult {
    /// Top-k hits sorted by descending BM25 score, then document id.
    pub hits: Vec<(DocId, f32)>,
    /// Traversal diagnostics for the search.
    pub stats: RawBm25FileSearchStats,
}

impl RawBm25CorpusStats {
    /// Create corpus stats from a document count, average document length, and
    /// document frequencies for terms that may be scored.
    ///
    /// A term that has postings in a searched segment must be present in `dfs`.
    /// Terms absent from all searched segments do not need entries.
    pub fn new<I>(num_docs: u32, avg_doc_len: f32, dfs: I) -> Self
    where
        I: IntoIterator<Item = (RawTermId, u32)>,
    {
        Self {
            num_docs,
            avg_doc_len,
            dfs: dfs.into_iter().collect(),
        }
    }

    /// Build query-scoped corpus stats from file-backed raw segments.
    ///
    /// This reads fixed segment directories for the unique query terms, not full
    /// postings payloads. Use this when querying several immutable raw files as
    /// one corpus.
    pub fn from_raw_files(
        segments: &mut [&mut RawSegmentFile],
        query_terms: &[RawTermId],
    ) -> Result<Self, RawScoringError<RawSegmentFileError>> {
        if query_terms.is_empty() {
            return Err(RawScoringError::EmptyQuery);
        }

        let terms = raw_term_multiplicities(query_terms);
        let mut num_docs = 0u32;
        let mut total_doc_len = 0.0f64;
        let mut dfs = HashMap::with_capacity(terms.len());
        for term in &terms {
            dfs.insert(term.term_id, 0u32);
        }

        for segment in segments.iter_mut() {
            let segment_docs = segment.num_docs();
            num_docs = num_docs.saturating_add(segment_docs);
            total_doc_len += segment.avg_doc_len() as f64 * segment_docs as f64;

            for term in &terms {
                let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
                if let Some(total_df) = dfs.get_mut(&term.term_id) {
                    *total_df = total_df.saturating_add(df);
                }
            }
        }

        if num_docs == 0 {
            return Err(RawScoringError::EmptyIndex);
        }

        Ok(Self {
            num_docs,
            avg_doc_len: (total_doc_len / num_docs as f64) as f32,
            dfs,
        })
    }

    /// Build query-scoped corpus stats from file-backed raw segments and one
    /// live in-memory raw postings shard.
    ///
    /// This reads fixed segment directories for the unique query terms and uses
    /// the live shard's postings metadata. Use this when querying sealed raw
    /// generations plus a bounded in-memory shard as one BM25 corpus.
    pub fn from_raw_files_and_index(
        segments: &mut [&mut RawSegmentFile],
        live_index: &PostingsIndex<RawTermId, u32>,
        query_terms: &[RawTermId],
    ) -> Result<Self, RawScoringError<RawSegmentFileError>> {
        if query_terms.is_empty() {
            return Err(RawScoringError::EmptyQuery);
        }

        let terms = raw_term_multiplicities(query_terms);
        let mut num_docs = 0u32;
        let mut total_doc_len = 0.0f64;
        let mut dfs = HashMap::with_capacity(terms.len());
        for term in &terms {
            dfs.insert(term.term_id, 0u32);
        }

        for segment in segments.iter_mut() {
            let segment_docs = segment.num_docs();
            num_docs = num_docs.saturating_add(segment_docs);
            total_doc_len += segment.avg_doc_len() as f64 * segment_docs as f64;

            for term in &terms {
                let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
                if let Some(total_df) = dfs.get_mut(&term.term_id) {
                    *total_df = total_df.saturating_add(df);
                }
            }
        }

        let live_docs = live_index.num_docs();
        num_docs = num_docs.saturating_add(live_docs);
        total_doc_len += live_index.avg_doc_len() as f64 * live_docs as f64;
        for term in &terms {
            if let Some(total_df) = dfs.get_mut(&term.term_id) {
                *total_df = total_df.saturating_add(live_index.df(&term.term_id));
            }
        }

        if num_docs == 0 {
            return Err(RawScoringError::EmptyIndex);
        }

        Ok(Self {
            num_docs,
            avg_doc_len: (total_doc_len / num_docs as f64) as f32,
            dfs,
        })
    }

    /// Build corpus stats for all terms present in file-backed raw segments and
    /// one live in-memory raw postings shard.
    ///
    /// This reads fixed segment directories, not postings payloads, and folds in
    /// the live shard's current document frequencies. Use this when serving many
    /// queries over sealed raw files plus a bounded live shard.
    pub fn from_raw_files_and_index_all_terms(
        segments: &mut [&mut RawSegmentFile],
        live_index: &PostingsIndex<RawTermId, u32>,
    ) -> Result<Self, RawScoringError<RawSegmentFileError>> {
        let mut num_docs = 0u32;
        let mut total_doc_len = 0.0f64;
        let mut dfs = HashMap::new();

        for segment in segments.iter_mut() {
            let segment_docs = segment.num_docs();
            num_docs = num_docs.saturating_add(segment_docs);
            total_doc_len += segment.avg_doc_len() as f64 * segment_docs as f64;

            segment
                .for_each_term_meta(|term| {
                    let total_df = dfs.entry(term.term_id()).or_insert(0u32);
                    *total_df = total_df.saturating_add(term.df());
                })
                .map_err(RawSegmentFileError::from)
                .map_err(RawScoringError::Source)?;
        }

        let live_docs = live_index.num_docs();
        num_docs = num_docs.saturating_add(live_docs);
        total_doc_len += live_index.avg_doc_len() as f64 * live_docs as f64;
        for term_id in live_index.terms() {
            let total_df = dfs.entry(*term_id).or_insert(0u32);
            *total_df = total_df.saturating_add(live_index.df(term_id));
        }

        if num_docs == 0 {
            return Err(RawScoringError::EmptyIndex);
        }

        Ok(Self {
            num_docs,
            avg_doc_len: (total_doc_len / num_docs as f64) as f32,
            dfs,
        })
    }

    /// Build corpus stats for all terms present in file-backed raw segments.
    ///
    /// This reads fixed segment directories, not postings payloads. Use this
    /// when serving many queries over the same immutable segment set and the
    /// caller wants to compute BM25 corpus statistics once per generation.
    pub fn from_raw_files_all_terms(
        segments: &mut [&mut RawSegmentFile],
    ) -> Result<Self, RawScoringError<RawSegmentFileError>> {
        let mut num_docs = 0u32;
        let mut total_doc_len = 0.0f64;
        let mut dfs = HashMap::new();

        for segment in segments.iter_mut() {
            let segment_docs = segment.num_docs();
            num_docs = num_docs.saturating_add(segment_docs);
            total_doc_len += segment.avg_doc_len() as f64 * segment_docs as f64;

            segment
                .for_each_term_meta(|term| {
                    let total_df = dfs.entry(term.term_id()).or_insert(0u32);
                    *total_df = total_df.saturating_add(term.df());
                })
                .map_err(RawSegmentFileError::from)
                .map_err(RawScoringError::Source)?;
        }

        if num_docs == 0 {
            return Err(RawScoringError::EmptyIndex);
        }

        Ok(Self {
            num_docs,
            avg_doc_len: (total_doc_len / num_docs as f64) as f32,
            dfs,
        })
    }

    fn from_reader<S>(
        segment: &mut S,
        query_terms: &[RawTermId],
    ) -> Result<Self, RawScoringError<S::Error>>
    where
        S: RawSegmentRead,
    {
        if query_terms.is_empty() {
            return Err(RawScoringError::EmptyQuery);
        }

        let num_docs = segment.num_docs();
        if num_docs == 0 {
            return Err(RawScoringError::EmptyIndex);
        }

        let terms = raw_term_multiplicities(query_terms);
        let mut dfs = HashMap::with_capacity(terms.len());
        for term in terms {
            let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
            dfs.insert(term.term_id, df);
        }

        Ok(Self {
            num_docs,
            avg_doc_len: segment.avg_doc_len(),
            dfs,
        })
    }

    /// Number of documents in the corpus represented by these stats.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Average document length used for BM25 length normalization.
    pub fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    /// Document frequency for a raw term, if present in this stats object.
    pub fn df(&self, term_id: RawTermId) -> Option<u32> {
        self.dfs.get(&term_id).copied()
    }
}

trait RawSegmentRead {
    type Error;

    fn num_docs(&self) -> u32;
    fn max_doc_id(&self) -> DocId;
    fn avg_doc_len(&self) -> f32;
    fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error>;
    fn document_len(&mut self, doc_id: DocId) -> Result<Option<u32>, Self::Error>;
    fn for_each_document_len(&mut self, visit: impl FnMut(DocId, u32)) -> Result<(), Self::Error>;
    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error>;
    fn for_each_posting_with_document_len(
        &mut self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Self::Error>;

    fn doc_id_range(&mut self) -> Result<Option<(DocId, DocId)>, Self::Error> {
        let mut min_doc_id = None;
        self.for_each_document_len(|doc_id, _| {
            min_doc_id = Some(min_doc_id.map_or(doc_id, |min: DocId| min.min(doc_id)));
        })?;
        Ok(min_doc_id.map(|min_doc_id| (min_doc_id, self.max_doc_id())))
    }
}

impl RawSegmentRead for RawSegment<'_> {
    type Error = postings::raw::Error;

    fn num_docs(&self) -> u32 {
        RawSegment::num_docs(self)
    }

    fn max_doc_id(&self) -> DocId {
        RawSegment::meta(self).max_doc_id()
    }

    fn avg_doc_len(&self) -> f32 {
        RawSegment::avg_doc_len(self)
    }

    fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error> {
        RawSegment::df(self, term_id)
    }

    fn document_len(&mut self, doc_id: DocId) -> Result<Option<u32>, Self::Error> {
        RawSegment::document_len(self, doc_id)
    }

    fn for_each_document_len(&mut self, visit: impl FnMut(DocId, u32)) -> Result<(), Self::Error> {
        RawSegment::for_each_document_len(self, visit)
    }

    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
        RawSegment::postings(self, term_id)?.collect()
    }

    fn for_each_posting_with_document_len(
        &mut self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Self::Error> {
        RawSegment::for_each_posting_with_document_len(self, term_id, visit)
    }

    fn doc_id_range(&mut self) -> Result<Option<(DocId, DocId)>, Self::Error> {
        RawSegment::doc_id_range(self)
    }
}

impl RawSegmentRead for RawSegmentFile {
    type Error = RawSegmentFileError;

    fn num_docs(&self) -> u32 {
        RawSegmentFile::num_docs(self)
    }

    fn max_doc_id(&self) -> DocId {
        RawSegmentFile::meta(self).max_doc_id()
    }

    fn avg_doc_len(&self) -> f32 {
        RawSegmentFile::avg_doc_len(self)
    }

    fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error> {
        Ok(RawSegmentFile::df(self, term_id)?)
    }

    fn document_len(&mut self, doc_id: DocId) -> Result<Option<u32>, Self::Error> {
        Ok(RawSegmentFile::document_len(self, doc_id)?)
    }

    fn for_each_document_len(&mut self, visit: impl FnMut(DocId, u32)) -> Result<(), Self::Error> {
        Ok(RawSegmentFile::for_each_document_len(self, visit)?)
    }

    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
        RawSegmentFile::postings(self, term_id)
    }

    fn for_each_posting_with_document_len(
        &mut self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Self::Error> {
        RawSegmentFile::for_each_posting_with_document_len(self, term_id, visit)
    }

    fn doc_id_range(&mut self) -> Result<Option<(DocId, DocId)>, Self::Error> {
        Ok(RawSegmentFile::doc_id_range(self)?)
    }
}

impl RawSegmentRead for &PostingsIndex<RawTermId, u32> {
    type Error = Infallible;

    fn num_docs(&self) -> u32 {
        PostingsIndex::num_docs(self)
    }

    fn max_doc_id(&self) -> DocId {
        self.document_ids().max().unwrap_or(0)
    }

    fn avg_doc_len(&self) -> f32 {
        PostingsIndex::avg_doc_len(self)
    }

    fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error> {
        Ok(PostingsIndex::df(*self, &term_id))
    }

    fn document_len(&mut self, doc_id: DocId) -> Result<Option<u32>, Self::Error> {
        Ok(Some(PostingsIndex::document_len(self, doc_id)))
    }

    fn for_each_document_len(
        &mut self,
        mut visit: impl FnMut(DocId, u32),
    ) -> Result<(), Self::Error> {
        for doc_id in self.document_ids() {
            visit(doc_id, PostingsIndex::document_len(self, doc_id));
        }
        Ok(())
    }

    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
        Ok(self.postings_iter(&term_id).collect())
    }

    fn for_each_posting_with_document_len(
        &mut self,
        term_id: RawTermId,
        mut visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Self::Error> {
        for (doc_id, weight) in self.postings_iter(&term_id) {
            visit(doc_id, weight, PostingsIndex::document_len(self, doc_id));
        }
        Ok(())
    }

    fn doc_id_range(&mut self) -> Result<Option<(DocId, DocId)>, Self::Error> {
        let mut ids = self.document_ids();
        let Some(first) = ids.next() else {
            return Ok(None);
        };
        let (min_doc_id, max_doc_id) = ids.fold((first, first), |(min_id, max_id), doc_id| {
            (min_id.min(doc_id), max_id.max(doc_id))
        });
        Ok(Some((min_doc_id, max_doc_id)))
    }
}

/// Retrieve top-k documents using BM25 over a byte-backed raw segment.
///
/// Query terms are numeric term ids from the caller's lexicon. Duplicate query
/// terms preserve BM25 scoring weight.
pub fn retrieve_bm25_raw_segment(
    segment: &RawSegment<'_>,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
) -> Result<Vec<(DocId, f32)>, RawScoringError<postings::raw::Error>> {
    let mut segment = *segment;
    let stats = RawBm25CorpusStats::from_reader(&mut segment, query_terms)?;
    retrieve_bm25_raw_with_stats(&mut segment, query_terms, k, params, &stats)
}

/// Retrieve top-k documents using BM25 over a byte-backed raw segment with
/// caller-provided corpus stats.
pub fn retrieve_bm25_raw_segment_with_stats(
    segment: &RawSegment<'_>,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<Vec<(DocId, f32)>, RawScoringError<postings::raw::Error>> {
    let mut segment = *segment;
    retrieve_bm25_raw_with_stats(&mut segment, query_terms, k, params, stats)
}

/// Retrieve top-k documents using BM25 over a file-backed raw segment.
///
/// The fixed segment directories stay in memory; posting payloads are read from
/// the file only for terms in the query.
pub fn retrieve_bm25_raw_file(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let stats = RawBm25CorpusStats::from_reader(segment, query_terms)?;
    retrieve_bm25_raw_with_stats(segment, query_terms, k, params, &stats)
}

/// Retrieve top-k documents using BM25 over a file-backed raw segment with
/// caller-provided corpus stats.
pub fn retrieve_bm25_raw_file_with_stats(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    retrieve_bm25_raw_file_with_stats_min_score(segment, query_terms, k, params, stats, None)
}

fn retrieve_bm25_raw_file_with_stats_min_score(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
    min_score: Option<f32>,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    retrieve_bm25_raw_file_with_search_stats_min_score(
        segment,
        query_terms,
        k,
        params,
        stats,
        min_score,
    )
    .map(|result| result.hits)
}

/// Retrieve top-k documents using BM25 over a file-backed raw segment and
/// return traversal diagnostics.
pub fn retrieve_bm25_raw_file_with_search_stats(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25FileSearchResult, RawScoringError<RawSegmentFileError>> {
    retrieve_bm25_raw_file_with_search_stats_min_score(segment, query_terms, k, params, stats, None)
}

fn retrieve_bm25_raw_file_with_search_stats_min_score(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
    min_score: Option<f32>,
) -> Result<RawBm25FileSearchResult, RawScoringError<RawSegmentFileError>> {
    if let Some(result) = retrieve_bm25_raw_file_with_stats_pruned_blocks(
        segment,
        query_terms,
        k,
        params,
        stats,
        min_score,
    )? {
        return Ok(result);
    }
    retrieve_bm25_raw_with_stats_and_search_stats(segment, query_terms, k, params, stats)
}

/// Retrieve top-k documents across file-backed raw segments as one corpus.
///
/// Segment document ids must already be globally unique. The helper builds
/// query-scoped corpus stats from all segments, scores each segment with those
/// stats, and merges the per-segment top-k lists.
pub fn retrieve_bm25_raw_files(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let stats = RawBm25CorpusStats::from_raw_files(segments, query_terms)?;
    retrieve_bm25_raw_files_with_stats(segments, query_terms, k, params, &stats)
}

/// Retrieve top-k documents across file-backed raw segments using
/// caller-provided corpus stats.
///
/// Segment document ids must already be globally unique. Passing shared corpus
/// stats keeps BM25 IDF and length normalization consistent across all
/// immutable segments.
pub fn retrieve_bm25_raw_files_with_stats(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    retrieve_bm25_raw_files_with_diagnostics_seeded(
        segments,
        query_terms,
        k,
        params,
        stats,
        Vec::new(),
    )
    .map(|result| result.hits)
}

/// Retrieve top-k documents across file-backed raw segments and one live
/// in-memory raw postings shard as one BM25 corpus.
///
/// Raw segment document ids and live-shard document ids must already be
/// globally unique among live documents. The caller owns any delete mask,
/// update policy, or manifest rule needed to keep those sets disjoint.
pub fn retrieve_bm25_raw_files_and_index(
    segments: &mut [&mut RawSegmentFile],
    live_index: &PostingsIndex<RawTermId, u32>,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let stats = RawBm25CorpusStats::from_raw_files_and_index(segments, live_index, query_terms)?;
    retrieve_bm25_raw_files_and_index_with_stats(
        segments,
        live_index,
        query_terms,
        k,
        params,
        &stats,
    )
}

/// Retrieve top-k documents across file-backed raw segments and one live
/// in-memory raw postings shard using caller-provided corpus stats.
///
/// Passing shared corpus stats keeps BM25 IDF and length normalization
/// consistent across sealed files and the live shard.
pub fn retrieve_bm25_raw_files_and_index_with_stats(
    segments: &mut [&mut RawSegmentFile],
    live_index: &PostingsIndex<RawTermId, u32>,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let mut live_hits = Vec::new();
    if live_index.num_docs() > 0 && k > 0 {
        let mut live_reader = live_index;
        live_hits = retrieve_bm25_raw_with_stats(&mut live_reader, query_terms, k, params, stats)
            .map_err(raw_infallible_error)?;
    }
    retrieve_bm25_raw_files_with_diagnostics_seeded(
        segments,
        query_terms,
        k,
        params,
        stats,
        live_hits,
    )
    .map(|result| result.hits)
}

/// Retrieve top-k documents across sealed raw segment files plus one live
/// in-memory raw postings shard and return segment and file-traversal
/// diagnostics for the sealed-file side.
///
/// Raw segment document ids and live-shard document ids must already be
/// globally unique among live documents. Passing shared corpus stats keeps
/// BM25 IDF and length normalization consistent across sealed files and the
/// live shard.
pub fn retrieve_bm25_raw_files_and_index_with_diagnostics(
    segments: &mut [&mut RawSegmentFile],
    live_index: &PostingsIndex<RawTermId, u32>,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25DiagnosticSearchResult, RawScoringError<RawSegmentFileError>> {
    let mut live_hits = Vec::new();
    if live_index.num_docs() > 0 && k > 0 {
        let mut live_reader = live_index;
        live_hits = retrieve_bm25_raw_with_stats(&mut live_reader, query_terms, k, params, stats)
            .map_err(raw_infallible_error)?;
    }
    retrieve_bm25_raw_files_with_diagnostics_seeded(
        segments,
        query_terms,
        k,
        params,
        stats,
        live_hits,
    )
}

/// Retrieve top-k documents across file-backed raw segments and return
/// segment-pruning diagnostics for the search.
///
/// Segment document ids must already be globally unique. Passing shared corpus
/// stats keeps BM25 IDF and length normalization consistent across all
/// immutable segments.
pub fn retrieve_bm25_raw_files_with_search_stats(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25SearchResult, RawScoringError<RawSegmentFileError>> {
    let result = retrieve_bm25_raw_files_with_diagnostics(segments, query_terms, k, params, stats)?;
    Ok(RawBm25SearchResult {
        hits: result.hits,
        stats: result.diagnostics.segments,
    })
}

/// Retrieve top-k documents across file-backed raw segments and return
/// segment and file-traversal diagnostics for the search.
///
/// Segment document ids must already be globally unique. Passing shared corpus
/// stats keeps BM25 IDF and length normalization consistent across all
/// immutable segments.
pub fn retrieve_bm25_raw_files_with_diagnostics(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25DiagnosticSearchResult, RawScoringError<RawSegmentFileError>> {
    retrieve_bm25_raw_files_with_diagnostics_seeded(
        segments,
        query_terms,
        k,
        params,
        stats,
        Vec::new(),
    )
}

fn retrieve_bm25_raw_files_with_diagnostics_seeded(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
    mut candidates: Vec<(DocId, f32)>,
) -> Result<RawBm25DiagnosticSearchResult, RawScoringError<RawSegmentFileError>> {
    if query_terms.is_empty() {
        return Err(RawScoringError::EmptyQuery);
    }
    if stats.num_docs() == 0 {
        return Err(RawScoringError::EmptyIndex);
    }
    let mut diagnostics = RawBm25SearchDiagnostics {
        segments: RawBm25SearchStats {
            segments_seen: segments.len(),
            ..RawBm25SearchStats::default()
        },
        ..RawBm25SearchDiagnostics::default()
    };
    if k == 0 {
        return Ok(RawBm25DiagnosticSearchResult {
            hits: Vec::new(),
            diagnostics,
        });
    }

    let avg_doc_len = stats.avg_doc_len();
    if avg_doc_len <= 0.0 || !avg_doc_len.is_finite() {
        return Ok(RawBm25DiagnosticSearchResult {
            hits: Vec::new(),
            diagnostics,
        });
    }

    if !candidates.is_empty() {
        candidates = top_k_positive_scored_docs(candidates, k);
    }
    candidates.reserve(k.saturating_mul(segments.len()));
    let terms = raw_term_multiplicities(query_terms);
    let mut order = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let mut upper_bound = 0.0;
        for term in &terms {
            let max_tf = segment
                .max_weight(term.term_id)
                .map_err(RawSegmentFileError::from)
                .map_err(RawScoringError::Source)?;
            if max_tf == 0 {
                continue;
            }
            let corpus_df = stats
                .df(term.term_id)
                .ok_or(RawScoringError::MissingCorpusStats(term.term_id))?;
            if corpus_df == 0 {
                continue;
            }
            let idf = bm25_idf_plus1(stats.num_docs(), corpus_df);
            upper_bound +=
                idf * term.count as f32 * bm25_tf_upper_bound(max_tf as f32, avg_doc_len, params);
        }
        if upper_bound > 0.0 || !upper_bound.is_finite() {
            order.push((index, upper_bound));
        } else {
            diagnostics.segments.segments_pruned += 1;
        }
    }
    order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut threshold = if candidates.len() >= k {
        candidates.last().map_or(0.0, |(_, score)| *score)
    } else {
        0.0
    };
    for (index, upper_bound) in order {
        if candidates.len() >= k && upper_bound < threshold {
            diagnostics.segments.segments_pruned += 1;
            continue;
        }
        diagnostics.segments.segments_scored += 1;
        let min_score = (candidates.len() >= k).then_some(threshold);
        let result = retrieve_bm25_raw_file_with_search_stats_min_score(
            segments[index],
            query_terms,
            k,
            params,
            stats,
            min_score,
        )?;
        diagnostics.add_file_stats(result.stats);
        candidates.extend(result.hits);
        if candidates.len() >= k {
            candidates = top_k_positive_scored_docs(candidates, k);
            threshold = candidates.last().map_or(0.0, |(_, score)| *score);
        }
    }

    Ok(RawBm25DiagnosticSearchResult {
        hits: top_k_positive_scored_docs(candidates, k),
        diagnostics,
    })
}

fn retrieve_bm25_raw_with_stats<S>(
    segment: &mut S,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<Vec<(DocId, f32)>, RawScoringError<S::Error>>
where
    S: RawSegmentRead,
{
    retrieve_bm25_raw_with_stats_and_search_stats(segment, query_terms, k, params, stats)
        .map(|result| result.hits)
}

fn raw_infallible_error<E>(err: RawScoringError<Infallible>) -> RawScoringError<E> {
    match err {
        RawScoringError::EmptyQuery => RawScoringError::EmptyQuery,
        RawScoringError::EmptyIndex => RawScoringError::EmptyIndex,
        RawScoringError::MissingCorpusStats(term_id) => {
            RawScoringError::MissingCorpusStats(term_id)
        }
        RawScoringError::Source(source) => match source {},
    }
}

fn retrieve_bm25_raw_with_stats_and_search_stats<S>(
    segment: &mut S,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25FileSearchResult, RawScoringError<S::Error>>
where
    S: RawSegmentRead,
{
    if query_terms.is_empty() {
        return Err(RawScoringError::EmptyQuery);
    }
    if stats.num_docs() == 0 {
        return Err(RawScoringError::EmptyIndex);
    }
    if k == 0 {
        return Ok(RawBm25FileSearchResult {
            hits: Vec::new(),
            stats: RawBm25FileSearchStats::default(),
        });
    }
    let avg_doc_len = stats.avg_doc_len();
    if avg_doc_len <= 0.0 || !avg_doc_len.is_finite() {
        return Ok(RawBm25FileSearchResult {
            hits: Vec::new(),
            stats: RawBm25FileSearchStats::default(),
        });
    }

    let mut terms = raw_term_multiplicities(query_terms);
    let mut touched_upper_bound = 0usize;
    for term in &mut terms {
        let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
        if df == 0 {
            continue;
        }
        let corpus_df = stats
            .df(term.term_id)
            .ok_or(RawScoringError::MissingCorpusStats(term.term_id))?;
        term.idf = bm25_idf_plus1(stats.num_docs(), corpus_df);
        if term.idf != 0.0 {
            touched_upper_bound = touched_upper_bound.saturating_add(df as usize);
        }
    }
    terms.retain(|term| term.idf != 0.0);
    if terms.is_empty() {
        return Ok(RawBm25FileSearchResult {
            hits: Vec::new(),
            stats: RawBm25FileSearchStats::default(),
        });
    }

    if let Some((doc_base, slots)) =
        dense_bm25_range(segment, touched_upper_bound).map_err(RawScoringError::Source)?
    {
        let stream_postings = should_stream_bm25(terms.len(), touched_upper_bound);
        let dense_plan = DenseRawBm25Plan {
            stream_postings,
            doc_base,
            slots,
            prefill_doc_lengths: should_prefill_dense_doc_lengths(
                segment.num_docs(),
                touched_upper_bound,
            ),
        };
        let stats = RawBm25FileSearchStats {
            path: if stream_postings {
                RawBm25FileSearchPath::DenseStream
            } else {
                RawBm25FileSearchPath::DenseCachedLengths
            },
            terms_scored: terms.len(),
            touched_postings_upper_bound: touched_upper_bound,
            dense_slots: slots,
            ..RawBm25FileSearchStats::default()
        };
        return retrieve_bm25_raw_dense(segment, dense_plan, terms, k, params, avg_doc_len)
            .map(|hits| RawBm25FileSearchResult { hits, stats });
    }

    let stream_postings = should_stream_bm25(terms.len(), touched_upper_bound);
    let search_path = if stream_postings {
        RawBm25FileSearchPath::SparseStream
    } else {
        RawBm25FileSearchPath::SparseCachedLengths
    };
    let search_stats = RawBm25FileSearchStats {
        path: search_path,
        terms_scored: terms.len(),
        touched_postings_upper_bound: touched_upper_bound,
        ..RawBm25FileSearchStats::default()
    };
    let capacity = touched_upper_bound.min(stats.num_docs() as usize);
    let mut scores = HashMap::with_capacity(capacity);
    if stream_postings {
        for term in terms {
            if term.idf == 0.0 {
                continue;
            }
            segment
                .for_each_posting_with_document_len(term.term_id, |doc_id, tf, doc_length| {
                    if doc_length == 0 {
                        return;
                    }

                    let tf_score = bm25_tf_score(tf as f32, doc_length as f32, avg_doc_len, params);
                    let contribution = term.idf * tf_score;
                    if contribution != 0.0 {
                        let score = scores.entry(doc_id).or_insert(0.0);
                        *score += contribution * term.count as f32;
                    }
                })
                .map_err(RawScoringError::Source)?;
        }
    } else {
        let mut doc_lengths = HashMap::with_capacity(capacity);
        for term in terms {
            if term.idf == 0.0 {
                continue;
            }
            for (doc_id, tf) in segment
                .postings(term.term_id)
                .map_err(RawScoringError::Source)?
            {
                let doc_length = match doc_lengths.get(&doc_id) {
                    Some(&len) => len,
                    None => {
                        let len = segment
                            .document_len(doc_id)
                            .map_err(RawScoringError::Source)?
                            .unwrap_or(0);
                        doc_lengths.insert(doc_id, len);
                        len
                    }
                };
                if doc_length == 0 {
                    continue;
                }

                let tf_score = bm25_tf_score(tf as f32, doc_length as f32, avg_doc_len, params);
                let contribution = term.idf * tf_score;
                if contribution != 0.0 {
                    let score = scores.entry(doc_id).or_insert(0.0);
                    *score += contribution * term.count as f32;
                }
            }
        }
    }

    Ok(RawBm25FileSearchResult {
        hits: top_k_positive_scored_docs(scores, k),
        stats: search_stats,
    })
}

const RAW_BM25_BLOCK_PRUNE_MIN_BLOCKS: usize = 16;
const RAW_BM25_BLOCK_PRUNE_MIN_TOUCHES: usize = 4096;

struct RawBm25BlockScoringPlan {
    terms: Vec<RawBm25BlockScoringTerm>,
    touched_upper_bound: usize,
    total_blocks: usize,
}

struct RawBm25PendingBlockTerm {
    term_id: RawTermId,
    count: usize,
    idf: f32,
}

struct RawBm25BlockScoringTerm {
    term_id: RawTermId,
    count: usize,
    idf: f32,
    blocks: Vec<RawPostingBlockMeta>,
}

#[derive(Clone, Copy)]
struct RawBm25BlockPruneContext {
    k: usize,
    params: Bm25Params,
    avg_doc_len: f32,
    min_score: Option<f32>,
}

enum RawBm25PrunedBlockSearch {
    Disabled,
    Empty,
    Search {
        plan: RawBm25BlockScoringPlan,
        avg_doc_len: f32,
    },
}

fn prepare_raw_bm25_pruned_block_search(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<RawBm25PrunedBlockSearch, RawScoringError<RawSegmentFileError>> {
    if query_terms.is_empty() {
        return Err(RawScoringError::EmptyQuery);
    }
    if stats.num_docs() == 0 {
        return Err(RawScoringError::EmptyIndex);
    }
    if k == 0 {
        return Ok(RawBm25PrunedBlockSearch::Empty);
    }
    let avg_doc_len = stats.avg_doc_len();
    if avg_doc_len <= 0.0 || !avg_doc_len.is_finite() {
        return Ok(RawBm25PrunedBlockSearch::Empty);
    }
    if !can_prune_raw_bm25_blocks(params) {
        return Ok(RawBm25PrunedBlockSearch::Disabled);
    }
    if (segment.num_docs() as usize) < RAW_BM25_BLOCK_PRUNE_MIN_TOUCHES {
        return Ok(RawBm25PrunedBlockSearch::Disabled);
    }

    let Some(plan) = prepare_raw_bm25_block_scoring_plan(segment, query_terms, stats)? else {
        return Ok(RawBm25PrunedBlockSearch::Disabled);
    };
    if plan.terms.is_empty() {
        return Ok(RawBm25PrunedBlockSearch::Empty);
    }
    if plan.total_blocks < RAW_BM25_BLOCK_PRUNE_MIN_BLOCKS || plan.total_blocks <= plan.terms.len()
    {
        return Ok(RawBm25PrunedBlockSearch::Disabled);
    }

    Ok(RawBm25PrunedBlockSearch::Search { plan, avg_doc_len })
}

fn retrieve_bm25_raw_file_with_stats_pruned_blocks(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
    min_score: Option<f32>,
) -> Result<Option<RawBm25FileSearchResult>, RawScoringError<RawSegmentFileError>> {
    let (plan, avg_doc_len) =
        match prepare_raw_bm25_pruned_block_search(segment, query_terms, k, params, stats)? {
            RawBm25PrunedBlockSearch::Disabled => return Ok(None),
            RawBm25PrunedBlockSearch::Empty => {
                return Ok(Some(RawBm25FileSearchResult {
                    hits: Vec::new(),
                    stats: RawBm25FileSearchStats::default(),
                }))
            }
            RawBm25PrunedBlockSearch::Search { plan, avg_doc_len } => (plan, avg_doc_len),
        };

    if let Some((doc_base, slots)) =
        dense_bm25_range(segment, plan.touched_upper_bound).map_err(RawScoringError::Source)?
    {
        let context = RawBm25BlockPruneContext {
            k,
            params,
            avg_doc_len,
            min_score,
        };
        let mut search_stats = RawBm25FileSearchStats {
            path: RawBm25FileSearchPath::BlockPrunedDense,
            terms_scored: plan.terms.len(),
            touched_postings_upper_bound: plan.touched_upper_bound,
            dense_slots: slots,
            ..RawBm25FileSearchStats::default()
        };
        return retrieve_bm25_raw_file_pruned_blocks_dense(
            segment,
            &plan.terms,
            (doc_base, slots),
            context,
            Some(&mut search_stats),
        )
        .map(|hits| {
            Some(RawBm25FileSearchResult {
                hits,
                stats: search_stats,
            })
        });
    }

    let context = RawBm25BlockPruneContext {
        k,
        params,
        avg_doc_len,
        min_score,
    };
    let mut search_stats = RawBm25FileSearchStats {
        path: RawBm25FileSearchPath::BlockPrunedSparse,
        terms_scored: plan.terms.len(),
        touched_postings_upper_bound: plan.touched_upper_bound,
        ..RawBm25FileSearchStats::default()
    };
    retrieve_bm25_raw_file_pruned_blocks_sparse(
        segment,
        &plan.terms,
        plan.touched_upper_bound.min(stats.num_docs() as usize),
        context,
        Some(&mut search_stats),
    )
    .map(|hits| {
        Some(RawBm25FileSearchResult {
            hits,
            stats: search_stats,
        })
    })
}

fn prepare_raw_bm25_block_scoring_plan(
    segment: &mut RawSegmentFile,
    query_terms: &[RawTermId],
    stats: &RawBm25CorpusStats,
) -> Result<Option<RawBm25BlockScoringPlan>, RawScoringError<RawSegmentFileError>> {
    let terms = raw_term_multiplicities(query_terms);
    let mut pending_terms = Vec::with_capacity(terms.len());
    let mut touched_upper_bound = 0usize;

    for term in terms {
        let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
        if df == 0 {
            continue;
        }
        let corpus_df = stats
            .df(term.term_id)
            .ok_or(RawScoringError::MissingCorpusStats(term.term_id))?;
        let idf = bm25_idf_plus1(stats.num_docs(), corpus_df);
        if idf == 0.0 {
            continue;
        }

        touched_upper_bound = touched_upper_bound.saturating_add(df as usize);
        pending_terms.push(RawBm25PendingBlockTerm {
            term_id: term.term_id,
            count: term.count,
            idf,
        });
    }

    if pending_terms.is_empty() {
        return Ok(Some(RawBm25BlockScoringPlan {
            terms: Vec::new(),
            touched_upper_bound,
            total_blocks: 0,
        }));
    }
    if touched_upper_bound < RAW_BM25_BLOCK_PRUNE_MIN_TOUCHES {
        return Ok(None);
    }

    let mut scoring_terms = Vec::with_capacity(pending_terms.len());
    let mut total_blocks = 0usize;
    for term in pending_terms {
        let blocks = segment
            .posting_blocks(term.term_id)
            .map_err(RawSegmentFileError::from)
            .map_err(RawScoringError::Source)?;
        if blocks.is_empty() {
            return Ok(None);
        }
        total_blocks = total_blocks.saturating_add(blocks.len());
        scoring_terms.push(RawBm25BlockScoringTerm {
            term_id: term.term_id,
            count: term.count,
            idf: term.idf,
            blocks,
        });
    }

    Ok(Some(RawBm25BlockScoringPlan {
        terms: scoring_terms,
        touched_upper_bound,
        total_blocks,
    }))
}

fn retrieve_bm25_raw_file_pruned_blocks_dense(
    segment: &mut RawSegmentFile,
    terms: &[RawBm25BlockScoringTerm],
    dense_range: (DocId, usize),
    context: RawBm25BlockPruneContext,
    mut stats: Option<&mut RawBm25FileSearchStats>,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let (doc_base, slots) = dense_range;
    let mut scores = vec![0.0; slots];
    let mut seen = vec![false; slots];
    let mut touched = Vec::new();
    let mut threshold = RawBm25TopKThreshold::new(context.k);

    for term in terms {
        for (block_index, &block) in term.blocks.iter().enumerate() {
            if let Some(stats) = stats.as_mut() {
                stats.term_blocks_seen += 1;
            }
            let upper_bound =
                raw_bm25_block_range_upper_bound(block, terms, context.avg_doc_len, context.params);
            if threshold
                .prune_threshold(context.min_score)
                .is_some_and(|threshold| upper_bound < threshold)
            {
                if let Some(stats) = stats.as_mut() {
                    stats.term_blocks_pruned += 1;
                }
                continue;
            }
            if let Some(stats) = stats.as_mut() {
                stats.term_blocks_scored += 1;
            }

            segment
                .for_each_posting_block_with_document_len(
                    term.term_id,
                    block_index as u32,
                    |doc_id, tf, doc_length| {
                        let slot = dense_doc_slot(doc_id, doc_base, slots);
                        if !seen[slot] {
                            seen[slot] = true;
                            touched.push((doc_id, slot));
                        }
                        if doc_length == 0 {
                            return;
                        }

                        let tf_score = bm25_tf_score(
                            tf as f32,
                            doc_length as f32,
                            context.avg_doc_len,
                            context.params,
                        );
                        let contribution = term.idf * tf_score;
                        if contribution != 0.0 {
                            scores[slot] += contribution * term.count as f32;
                            threshold.update(doc_id, scores[slot]);
                        }
                    },
                )
                .map_err(RawScoringError::Source)?;
        }
    }

    Ok(top_k_positive_scored_docs(
        touched
            .into_iter()
            .map(|(doc_id, slot)| (doc_id, scores[slot])),
        context.k,
    ))
}

fn retrieve_bm25_raw_file_pruned_blocks_sparse(
    segment: &mut RawSegmentFile,
    terms: &[RawBm25BlockScoringTerm],
    capacity: usize,
    context: RawBm25BlockPruneContext,
    mut stats: Option<&mut RawBm25FileSearchStats>,
) -> Result<Vec<(DocId, f32)>, RawScoringError<RawSegmentFileError>> {
    let mut scores = HashMap::with_capacity(capacity);
    let mut threshold = RawBm25TopKThreshold::new(context.k);

    for term in terms {
        for (block_index, &block) in term.blocks.iter().enumerate() {
            if let Some(stats) = stats.as_mut() {
                stats.term_blocks_seen += 1;
            }
            let upper_bound =
                raw_bm25_block_range_upper_bound(block, terms, context.avg_doc_len, context.params);
            if threshold
                .prune_threshold(context.min_score)
                .is_some_and(|threshold| upper_bound < threshold)
            {
                if let Some(stats) = stats.as_mut() {
                    stats.term_blocks_pruned += 1;
                }
                continue;
            }
            if let Some(stats) = stats.as_mut() {
                stats.term_blocks_scored += 1;
            }

            segment
                .for_each_posting_block_with_document_len(
                    term.term_id,
                    block_index as u32,
                    |doc_id, tf, doc_length| {
                        if doc_length == 0 {
                            return;
                        }

                        let tf_score = bm25_tf_score(
                            tf as f32,
                            doc_length as f32,
                            context.avg_doc_len,
                            context.params,
                        );
                        let contribution = term.idf * tf_score;
                        if contribution != 0.0 {
                            let score = scores.entry(doc_id).or_insert(0.0);
                            *score += contribution * term.count as f32;
                            threshold.update(doc_id, *score);
                        }
                    },
                )
                .map_err(RawScoringError::Source)?;
        }
    }

    Ok(top_k_positive_scored_docs(scores, context.k))
}

fn raw_bm25_block_range_upper_bound(
    block: RawPostingBlockMeta,
    terms: &[RawBm25BlockScoringTerm],
    avg_doc_len: f32,
    params: Bm25Params,
) -> f32 {
    terms
        .iter()
        .map(|term| {
            let max_tf = max_overlapping_raw_block_weight(&term.blocks, block);
            if max_tf == 0 {
                return 0.0;
            }
            term.idf * term.count as f32 * bm25_tf_upper_bound(max_tf as f32, avg_doc_len, params)
        })
        .sum()
}

fn max_overlapping_raw_block_weight(
    blocks: &[RawPostingBlockMeta],
    target: RawPostingBlockMeta,
) -> u32 {
    let start = blocks.partition_point(|block| block.last_doc_id() < target.base_doc_id());
    let mut max_weight = 0u32;
    for block in &blocks[start..] {
        if block.base_doc_id() > target.last_doc_id() {
            break;
        }
        max_weight = max_weight.max(block.max_weight());
    }
    max_weight
}

fn can_prune_raw_bm25_blocks(params: Bm25Params) -> bool {
    params.k1.is_finite()
        && params.k1 >= 0.0
        && params.b.is_finite()
        && (0.0..=1.0).contains(&params.b)
        && match params.variant {
            Bm25Variant::Standard => true,
            Bm25Variant::BM25L { delta } | Bm25Variant::BM25Plus { delta } => {
                delta.is_finite() && delta >= 0.0
            }
        }
}

struct RawBm25TopKThreshold {
    ranked: Vec<(DocId, f32)>,
    k: usize,
    sorted: bool,
}

impl RawBm25TopKThreshold {
    fn new(k: usize) -> Self {
        Self {
            ranked: Vec::with_capacity(k),
            k,
            sorted: false,
        }
    }

    fn update(&mut self, doc_id: DocId, score: f32) {
        if self.k == 0 || !score.is_finite() || score <= 0.0 {
            return;
        }

        if let Some(index) = self
            .ranked
            .iter()
            .position(|(ranked_doc_id, _)| *ranked_doc_id == doc_id)
        {
            self.ranked[index].1 = score;
            if self.sorted {
                self.bubble_up(index);
            }
            return;
        }

        if self.ranked.len() < self.k {
            self.ranked.push((doc_id, score));
            self.sorted = false;
            return;
        }

        self.sort_if_needed();
        let candidate = (doc_id, score);
        if cmp_raw_bm25_doc_scores(
            &candidate,
            self.ranked.last().expect("top-k buffer is full"),
        )
        .is_lt()
        {
            let last = self.ranked.len() - 1;
            self.ranked[last] = candidate;
            self.bubble_up(last);
        }
    }

    fn threshold(&mut self) -> Option<f32> {
        if self.ranked.len() < self.k {
            return None;
        }
        self.sort_if_needed();
        self.ranked.last().map(|(_, score)| *score)
    }

    fn prune_threshold(&mut self, min_score: Option<f32>) -> Option<f32> {
        let local = self.threshold();
        let min_score = min_score.filter(|score| score.is_finite() && *score > 0.0);
        match (local, min_score) {
            (Some(local), Some(min_score)) => Some(local.max(min_score)),
            (Some(local), None) => Some(local),
            (None, Some(min_score)) => Some(min_score),
            (None, None) => None,
        }
    }

    fn sort_if_needed(&mut self) {
        if !self.sorted {
            self.ranked.sort_by(cmp_raw_bm25_doc_scores);
            self.sorted = true;
        }
    }

    fn bubble_up(&mut self, mut index: usize) {
        while index > 0
            && cmp_raw_bm25_doc_scores(&self.ranked[index], &self.ranked[index - 1]).is_lt()
        {
            self.ranked.swap(index, index - 1);
            index -= 1;
        }
    }
}

#[inline]
fn cmp_raw_bm25_doc_scores(a: &(DocId, f32), b: &(DocId, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

struct DenseRawBm25Plan {
    stream_postings: bool,
    doc_base: DocId,
    slots: usize,
    prefill_doc_lengths: bool,
}

fn retrieve_bm25_raw_dense<S>(
    segment: &mut S,
    plan: DenseRawBm25Plan,
    terms: Vec<WeightedRawTerm>,
    k: usize,
    params: Bm25Params,
    avg_doc_len: f32,
) -> Result<Vec<(DocId, f32)>, RawScoringError<S::Error>>
where
    S: RawSegmentRead,
{
    let doc_base = plan.doc_base;
    let slots = plan.slots;
    let mut scores = vec![0.0; slots];
    let mut doc_lengths = (!plan.stream_postings).then(|| vec![0u32; slots]);
    let mut seen = vec![false; slots];
    let mut touched = Vec::new();

    if plan.stream_postings {
        for term in terms {
            if term.idf == 0.0 {
                continue;
            }
            segment
                .for_each_posting_with_document_len(term.term_id, |doc_id, tf, doc_length| {
                    let slot = dense_doc_slot(doc_id, doc_base, slots);
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push((doc_id, slot));
                    }

                    if doc_length == 0 {
                        return;
                    }

                    let tf_score = bm25_tf_score(tf as f32, doc_length as f32, avg_doc_len, params);
                    let contribution = term.idf * tf_score;
                    if contribution != 0.0 {
                        scores[slot] += contribution * term.count as f32;
                    }
                })
                .map_err(RawScoringError::Source)?;
        }
    } else {
        let doc_lengths = doc_lengths
            .as_mut()
            .expect("dense doc-length cache exists for non-streaming path");
        if plan.prefill_doc_lengths {
            segment
                .for_each_document_len(|doc_id, doc_length| {
                    let slot = dense_doc_slot(doc_id, doc_base, slots);
                    doc_lengths[slot] = doc_length;
                })
                .map_err(RawScoringError::Source)?;
        }
        for term in terms {
            if term.idf == 0.0 {
                continue;
            }
            for (doc_id, tf) in segment
                .postings(term.term_id)
                .map_err(RawScoringError::Source)?
            {
                let slot = dense_doc_slot(doc_id, doc_base, slots);
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push((doc_id, slot));
                }

                let mut doc_length = doc_lengths[slot];
                if doc_length == 0 {
                    doc_length = segment
                        .document_len(doc_id)
                        .map_err(RawScoringError::Source)?
                        .unwrap_or(0);
                    doc_lengths[slot] = doc_length;
                }
                if doc_length == 0 {
                    continue;
                }

                let tf_score = bm25_tf_score(tf as f32, doc_length as f32, avg_doc_len, params);
                let contribution = term.idf * tf_score;
                if contribution != 0.0 {
                    scores[slot] += contribution * term.count as f32;
                }
            }
        }
    }

    Ok(top_k_positive_scored_docs(
        touched
            .into_iter()
            .map(|(doc_id, slot)| (doc_id, scores[slot])),
        k,
    ))
}

fn should_prefill_dense_doc_lengths(num_docs: u32, touched_upper_bound: usize) -> bool {
    touched_upper_bound > 0 && touched_upper_bound >= (num_docs as usize).div_ceil(2)
}

const DENSE_BM25_MAX_SLOTS: usize = 1_000_000;
const STREAMING_BM25_MAX_QUERY_TERMS: usize = 8;
const STREAMING_BM25_MIN_TOUCHED_POSTINGS: usize = 1_000_000;

fn dense_bm25_range<S>(
    segment: &mut S,
    touched_upper_bound: usize,
) -> Result<Option<(DocId, usize)>, S::Error>
where
    S: RawSegmentRead,
{
    let Some((min_doc_id, max_doc_id)) = segment.doc_id_range()? else {
        return Ok(None);
    };
    let Some(span) = max_doc_id.checked_sub(min_doc_id) else {
        return Ok(None);
    };
    let Some(slots) = usize::try_from(span)
        .ok()
        .and_then(|span| span.checked_add(1))
    else {
        return Ok(None);
    };
    let useful_limit = touched_upper_bound.saturating_mul(2).max(4096);
    Ok((slots <= DENSE_BM25_MAX_SLOTS && slots <= useful_limit).then_some((min_doc_id, slots)))
}

#[inline]
fn dense_doc_slot(doc_id: DocId, doc_base: DocId, slots: usize) -> usize {
    let slot = doc_id.wrapping_sub(doc_base) as usize;
    debug_assert!(slot < slots);
    slot
}

fn should_stream_bm25(unique_terms: usize, touched_upper_bound: usize) -> bool {
    unique_terms <= STREAMING_BM25_MAX_QUERY_TERMS
        || touched_upper_bound >= STREAMING_BM25_MIN_TOUCHED_POSTINGS
}

fn bm25_tf_upper_bound(max_tf: f32, avg_doc_len: f32, params: Bm25Params) -> f32 {
    if max_tf <= 0.0 {
        return 0.0;
    }

    let saturated = params.k1.max(0.0) + 1.0;
    match params.variant {
        Bm25Variant::Standard => bm25_tf_score(max_tf, 0.0, avg_doc_len, params).min(saturated),
        Bm25Variant::BM25L { .. } => saturated,
        Bm25Variant::BM25Plus { delta } => {
            bm25_tf_score(max_tf, 0.0, avg_doc_len, params).min(saturated + delta.max(0.0))
        }
    }
}

#[derive(Debug)]
struct WeightedRawTerm {
    term_id: RawTermId,
    count: usize,
    idf: f32,
}

fn raw_term_multiplicities(query_terms: &[RawTermId]) -> Vec<WeightedRawTerm> {
    let mut terms = query_terms.to_vec();
    terms.sort_unstable();

    let mut counts: Vec<WeightedRawTerm> = Vec::with_capacity(terms.len());
    for term_id in terms {
        if let Some(last) = counts.last_mut() {
            if last.term_id == term_id {
                last.count += 1;
                continue;
            }
        }
        counts.push(WeightedRawTerm {
            term_id,
            count: 1,
            idf: 0.0,
        });
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::InvertedIndex;
    use postings::raw::{write_u64_u32_segment, RawDocument};
    use postings::PostingsIndex;

    fn build_raw_docs() -> Vec<Vec<(RawTermId, u32)>> {
        vec![
            vec![(10, 2), (20, 1)],
            vec![(10, 1), (30, 4)],
            vec![(20, 3), (30, 1)],
            vec![(40, 2)],
        ]
    }

    fn build_memory_index(raw_docs: &[Vec<(RawTermId, u32)>]) -> InvertedIndex {
        let mut index = InvertedIndex::new();
        for (doc_id, terms) in raw_docs.iter().enumerate() {
            add_raw_doc_to_memory_index(&mut index, doc_id as DocId, terms);
        }
        index
    }

    fn add_raw_doc_to_memory_index(
        index: &mut InvertedIndex,
        doc_id: DocId,
        terms: &[(RawTermId, u32)],
    ) {
        let mut expanded = Vec::new();
        for &(term_id, weight) in terms {
            for _ in 0..weight {
                expanded.push(term_id.to_string());
            }
        }
        index.add_document(doc_id, &expanded);
    }

    fn build_raw_bytes(raw_docs: &[Vec<(RawTermId, u32)>]) -> Vec<u8> {
        let docs: Vec<_> = raw_docs
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        write_u64_u32_segment(&docs).unwrap()
    }

    fn build_raw_bytes_with_doc_ids(raw_docs: &[(DocId, Vec<(RawTermId, u32)>)]) -> Vec<u8> {
        let docs: Vec<_> = raw_docs
            .iter()
            .map(|(doc_id, terms)| RawDocument::new(*doc_id, terms))
            .collect();
        write_u64_u32_segment(&docs).unwrap()
    }

    fn generated_raw_docs(
        n_docs: u32,
        vocab: RawTermId,
        terms_per_doc: usize,
        seed: u64,
        doc_id: impl Fn(u32) -> DocId,
    ) -> Vec<(DocId, Vec<(RawTermId, u32)>)> {
        let mut state = seed;
        (0..n_docs)
            .map(|i| {
                let terms = (0..terms_per_doc)
                    .map(|_| {
                        let term_id = zipf_term(&mut state, vocab);
                        let tf = 1 + (xorshift(&mut state) % 4) as u32;
                        (term_id, tf)
                    })
                    .collect();
                (doc_id(i), terms)
            })
            .collect()
    }

    fn build_memory_index_with_doc_ids(
        raw_docs: &[(DocId, Vec<(RawTermId, u32)>)],
    ) -> InvertedIndex {
        let mut index = InvertedIndex::new();
        for (doc_id, terms) in raw_docs {
            add_raw_doc_to_memory_index(&mut index, *doc_id, terms);
        }
        index
    }

    fn terms_by_df(index: &InvertedIndex, vocab: RawTermId, min_df: u32) -> Vec<RawTermId> {
        let mut terms: Vec<_> = (0..vocab)
            .filter(|term_id| index.doc_frequency(&term_id.to_string()) >= min_df)
            .collect();
        terms.sort_by_key(|term_id| std::cmp::Reverse(index.doc_frequency(&term_id.to_string())));
        terms
    }

    fn raw_bm25_memory_hits(
        index: &InvertedIndex,
        query: &[RawTermId],
        k: usize,
        params: Bm25Params,
    ) -> Vec<(DocId, f32)> {
        let memory_query: Vec<String> = query.iter().map(ToString::to_string).collect();
        index.retrieve(&memory_query, k, params).unwrap()
    }

    fn assert_hits_close(expected: &[(DocId, f32)], got: &[(DocId, f32)], context: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{context}: result count diverged: expected={expected:?}, got={got:?}"
        );
        for ((expected_doc, expected_score), (got_doc, got_score)) in expected.iter().zip(got) {
            let tolerance = 1e-5_f32.max(expected_score.abs() * 1e-6);
            assert_eq!(
                got_doc, expected_doc,
                "{context}: doc order diverged: expected={expected:?}, got={got:?}"
            );
            assert!(
                (expected_score - got_score).abs() <= tolerance,
                "{context}: score diverged for doc {expected_doc}: expected={expected_score}, got={got_score}, tolerance={tolerance}"
            );
        }
    }

    fn zipf_term(state: &mut u64, vocab: RawTermId) -> RawTermId {
        let u = rand_f64(state);
        let n = vocab as f64;
        ((u * (n + 1.0).ln()).exp() - 1.0) as RawTermId
    }

    fn rand_f64(state: &mut u64) -> f64 {
        (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn raw_term_dictionary_assigns_stable_insertion_ids() {
        let mut dictionary = RawTermDictionary::new();

        let search = dictionary.insert("search");
        let rust = dictionary.insert("rust");
        let duplicate = dictionary.insert(String::from("search"));

        assert_eq!(search, 0);
        assert_eq!(rust, 1);
        assert_eq!(duplicate, search);
        assert_eq!(dictionary.id("search"), Some(search));
        assert_eq!(dictionary.id("missing"), None);
        assert_eq!(dictionary.term(search), Some("search"));
        assert_eq!(dictionary.term(99), None);
        assert_eq!(
            dictionary.terms().collect::<Vec<_>>(),
            vec![(search, "search"), (rust, "rust")]
        );
    }

    #[test]
    fn raw_term_dictionary_sorted_constructor_is_order_independent() {
        let first = RawTermDictionary::from_terms_sorted(["beta", "alpha", "beta", "gamma"]);
        let second = RawTermDictionary::from_terms_sorted(["gamma", "beta", "alpha"]);

        assert_eq!(first, second);
        assert_eq!(first.id("alpha"), Some(0));
        assert_eq!(first.id("beta"), Some(1));
        assert_eq!(first.id("gamma"), Some(2));
        assert_eq!(
            first.terms().collect::<Vec<_>>(),
            vec![(0, "alpha"), (1, "beta"), (2, "gamma"),]
        );
    }

    #[test]
    fn raw_term_dictionary_loads_persisted_id_order() {
        let mut original = RawTermDictionary::new();
        original.insert("beta");
        original.insert("alpha");
        original.insert("gamma");
        let persisted_terms: Vec<_> = original.terms().map(|(_, term)| term.to_owned()).collect();

        let loaded = RawTermDictionary::from_terms_in_id_order(&persisted_terms).unwrap();

        assert_eq!(loaded, original);
        assert_eq!(loaded.id("beta"), Some(0));
        assert_eq!(loaded.id("alpha"), Some(1));
        assert_eq!(loaded.id("gamma"), Some(2));
    }

    #[test]
    fn raw_term_dictionary_rejects_duplicate_persisted_terms() {
        let err =
            RawTermDictionary::from_terms_in_id_order(["alpha", "beta", "alpha"]).unwrap_err();

        assert_eq!(err, RawTermDictionaryError::DuplicateTerm { term_id: 2 });
    }

    #[test]
    fn raw_term_dictionary_encodes_documents_and_queries() {
        let mut dictionary = RawTermDictionary::from_terms_sorted(["rust", "search"]);

        let document = dictionary
            .encode_document(["rust", "search", "rust", "new"])
            .unwrap();
        let query = dictionary.encode_query(["missing", "search", "search", "rust"]);

        assert_eq!(dictionary.id("rust"), Some(0));
        assert_eq!(dictionary.id("search"), Some(1));
        assert_eq!(dictionary.id("new"), Some(2));
        assert_eq!(document, vec![(0, 2), (1, 1), (2, 1)]);
        assert_eq!(query, vec![1, 1, 0]);
    }

    #[test]
    fn raw_term_dictionary_feeds_raw_bm25_segment() {
        let mut dictionary = RawTermDictionary::new();
        let encoded_docs = [
            dictionary
                .encode_document(["rust", "search", "search"])
                .unwrap(),
            dictionary.encode_document(["rust"]).unwrap(),
            dictionary.encode_document(["other"]).unwrap(),
        ];
        let raw_docs: Vec<_> = encoded_docs
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        let bytes = write_u64_u32_segment(&raw_docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();

        let query = dictionary.encode_query(["search", "missing", "search"]);
        assert_eq!(query, vec![dictionary.id("search").unwrap(); 2]);

        let hits = retrieve_bm25_raw_segment(&segment, &query, 3, Bm25Params::default()).unwrap();
        assert_eq!(hits.first().map(|(doc_id, _)| *doc_id), Some(0));
    }

    #[test]
    fn raw_bm25_segment_matches_in_memory_index() {
        let raw_docs = build_raw_docs();
        let index = build_memory_index(&raw_docs);
        let bytes = build_raw_bytes(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();

        let query = vec![10, 30, 10];
        let raw_hits =
            retrieve_bm25_raw_segment(&segment, &query, 10, Bm25Params::default()).unwrap();
        let memory_query: Vec<String> = query.iter().map(ToString::to_string).collect();
        let memory_hits = index
            .retrieve(&memory_query, 10, Bm25Params::default())
            .unwrap();

        assert_eq!(raw_hits.len(), memory_hits.len());
        for ((raw_doc, raw_score), (memory_doc, memory_score)) in
            raw_hits.iter().zip(memory_hits.iter())
        {
            assert_eq!(raw_doc, memory_doc);
            assert!(
                (raw_score - memory_score).abs() < 1e-6,
                "doc {raw_doc}: raw={raw_score}, memory={memory_score}"
            );
        }
    }

    #[test]
    fn raw_bm25_matches_in_memory_across_query_shapes_and_sparse_doc_ids() {
        let raw_docs = generated_raw_docs(320, 512, 40, 0xb025_5eed, |i| 1_000 + i * 7);
        let index = build_memory_index_with_doc_ids(&raw_docs);
        let bytes = build_raw_bytes_with_doc_ids(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        let terms = terms_by_df(&index, 512, 8);
        assert!(
            terms.len() >= 48,
            "fixture needs enough queryable terms, got {}",
            terms.len()
        );

        let params = [
            Bm25Params::default(),
            Bm25Params::bm25l(),
            Bm25Params::bm25plus(),
        ];
        for params in params {
            for query_len in [1usize, 4, 16, 48] {
                let mut query = terms[..query_len].to_vec();
                if query_len >= 4 {
                    query.push(query[0]);
                    query.push(query[2]);
                }
                for k in [1usize, 10, 50] {
                    let context = format!("query_len={query_len}, k={k}, params={params:?}");
                    let expected = raw_bm25_memory_hits(&index, &query, k, params);
                    let byte_hits = retrieve_bm25_raw_segment(&segment, &query, k, params).unwrap();
                    assert_hits_close(&expected, &byte_hits, &format!("{context}, byte"));
                    let file_hits =
                        retrieve_bm25_raw_file(&mut file_segment, &query, k, params).unwrap();
                    assert_hits_close(&expected, &file_hits, &format!("{context}, file"));
                }
            }
        }
    }

    #[test]
    fn raw_bm25_files_match_in_memory_across_query_shapes() {
        let raw_docs = generated_raw_docs(360, 512, 36, 0xf17e_b025, |i| 10_000 + i * 5);
        let index = build_memory_index_with_doc_ids(&raw_docs);
        let terms = terms_by_df(&index, 512, 8);
        assert!(
            terms.len() >= 32,
            "fixture needs enough queryable terms, got {}",
            terms.len()
        );

        let dir = tempfile::tempdir().unwrap();
        let mut files: Vec<_> = raw_docs
            .chunks(90)
            .enumerate()
            .map(|(chunk_id, chunk)| {
                let path = dir.path().join(format!("chunk-{chunk_id}.raw"));
                std::fs::write(&path, build_raw_bytes_with_doc_ids(chunk)).unwrap();
                RawSegmentFile::open(&path).unwrap()
            })
            .collect();

        for query_len in [2usize, 8, 32] {
            let query = terms[..query_len].to_vec();
            for k in [3usize, 10, 40] {
                let mut refs: Vec<_> = files.iter_mut().collect();
                let stats = RawBm25CorpusStats::from_raw_files(&mut refs, &query).unwrap();
                let got = retrieve_bm25_raw_files_with_stats(
                    &mut refs,
                    &query,
                    k,
                    Bm25Params::default(),
                    &stats,
                )
                .unwrap();
                let expected = raw_bm25_memory_hits(&index, &query, k, Bm25Params::default());
                assert_hits_close(
                    &expected,
                    &got,
                    &format!("multi-file query_len={query_len}, k={k}"),
                );
            }
        }
    }

    #[test]
    fn raw_bm25_file_matches_byte_segment() {
        let raw_docs = build_raw_docs();
        let bytes = build_raw_bytes(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let query = vec![10, 20, 30];

        let byte_hits =
            retrieve_bm25_raw_segment(&segment, &query, 3, Bm25Params::default()).unwrap();
        let file_hits =
            retrieve_bm25_raw_file(&mut file_segment, &query, 3, Bm25Params::default()).unwrap();

        assert_eq!(file_hits, byte_hits);
    }

    #[test]
    fn raw_bm25_file_block_pruning_uses_overlapping_term_bounds() {
        const TERM_COMMON: RawTermId = 10;
        const TERM_TAIL: RawTermId = 20;
        let raw_docs: Vec<_> = (0..4096)
            .map(|doc_id| {
                let mut terms = vec![(
                    TERM_COMMON,
                    if (3000..3010).contains(&doc_id) {
                        20
                    } else {
                        1
                    },
                )];
                if doc_id >= 3000 {
                    terms.push((TERM_TAIL, 90));
                }
                (doc_id, terms)
            })
            .collect();
        let bytes = build_raw_bytes_with_doc_ids(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let query = vec![TERM_COMMON, TERM_TAIL];
        let params = Bm25Params::default();
        let stats = {
            let mut segments = [&mut file_segment];
            RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap()
        };
        let expected =
            retrieve_bm25_raw_segment_with_stats(&segment, &query, 10, params, &stats).unwrap();

        let top_doc_id = expected[0].0;
        assert!(
            (3000..3128).contains(&top_doc_id),
            "fixture top doc {top_doc_id} should land in the overlapping tail block"
        );

        let plan = prepare_raw_bm25_block_scoring_plan(&mut file_segment, &query, &stats)
            .unwrap()
            .expect("fixture should use block plan");
        assert!(plan.total_blocks >= RAW_BM25_BLOCK_PRUNE_MIN_BLOCKS);
        let common = plan
            .terms
            .iter()
            .find(|term| term.term_id == TERM_COMMON)
            .unwrap();
        let target = common
            .blocks
            .iter()
            .copied()
            .find(|block| block.base_doc_id() <= top_doc_id && block.last_doc_id() >= top_doc_id)
            .unwrap();
        let bound =
            raw_bm25_block_range_upper_bound(target, &plan.terms, stats.avg_doc_len(), params);
        assert!(
            bound >= expected[0].1,
            "block upper bound {bound} must include overlapping term contributions \
             above realized top score {}",
            expected[0].1
        );

        let pruned = retrieve_bm25_raw_file_with_stats_pruned_blocks(
            &mut file_segment,
            &query,
            10,
            params,
            &stats,
            None,
        )
        .unwrap()
        .expect("fixture should enter the block-pruned path");
        assert_hits_close(&expected, &pruned.hits, "block-pruned file BM25");
        assert_eq!(pruned.stats.path, RawBm25FileSearchPath::BlockPrunedDense);
        assert_eq!(pruned.stats.terms_scored, plan.terms.len());
        assert_eq!(pruned.stats.term_blocks_seen, plan.total_blocks);
        assert!(
            pruned.stats.term_blocks_scored > 0,
            "fixture must decode at least one term block"
        );

        let file_hits =
            retrieve_bm25_raw_file_with_stats(&mut file_segment, &query, 10, params, &stats)
                .unwrap();
        assert_hits_close(&expected, &file_hits, "public file BM25");
    }

    #[test]
    fn raw_bm25_file_block_pruning_stats_report_skipped_blocks() {
        const TERM: RawTermId = 10;
        const FIRST_BLOCK_DOCS: u32 = 128;
        let raw_docs: Vec<_> = (0..4096)
            .map(|doc_id| {
                (
                    doc_id,
                    vec![(TERM, if doc_id < FIRST_BLOCK_DOCS { 10_000 } else { 1 })],
                )
            })
            .collect();
        let bytes = build_raw_bytes_with_doc_ids(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let query = vec![TERM];
        let params = Bm25Params::default();
        let stats = {
            let mut segments = [&mut file_segment];
            RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap()
        };
        let expected =
            retrieve_bm25_raw_segment_with_stats(&segment, &query, 10, params, &stats).unwrap();

        let result =
            retrieve_bm25_raw_file_with_search_stats(&mut file_segment, &query, 10, params, &stats)
                .unwrap();

        assert_hits_close(&expected, &result.hits, "block-pruned file BM25");
        assert_eq!(result.stats.path, RawBm25FileSearchPath::BlockPrunedDense);
        assert!(result.stats.term_blocks_seen >= RAW_BM25_BLOCK_PRUNE_MIN_BLOCKS);
        assert!(
            result.stats.term_blocks_pruned > 0,
            "fixture must skip at least one low-bound term block"
        );
        assert_eq!(
            result.stats.term_blocks_seen,
            result.stats.term_blocks_scored + result.stats.term_blocks_pruned
        );
    }

    #[test]
    fn raw_bm25_file_block_pruning_uses_seeded_score_floor() {
        const TERM: RawTermId = 10;
        const TOTAL_DOCS: u32 = 4096;
        const LAST_BLOCK_DOCS: u32 = 128;
        let high_start = TOTAL_DOCS - LAST_BLOCK_DOCS;
        let raw_docs: Vec<_> = (0..TOTAL_DOCS)
            .map(|doc_id| {
                (
                    doc_id,
                    vec![(TERM, if doc_id >= high_start { 10_000 } else { 1 })],
                )
            })
            .collect();
        let bytes = build_raw_bytes_with_doc_ids(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let query = vec![TERM];
        let params = Bm25Params::default();
        let stats = {
            let mut segments = [&mut file_segment];
            RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap()
        };
        let plan = match prepare_raw_bm25_pruned_block_search(
            &mut file_segment,
            &query,
            10,
            params,
            &stats,
        )
        .unwrap()
        {
            RawBm25PrunedBlockSearch::Search { plan, .. } => plan,
            _ => panic!("fixture should enter the block-pruned path"),
        };
        let low_bound = raw_bm25_block_range_upper_bound(
            plan.terms[0].blocks[0],
            &plan.terms,
            stats.avg_doc_len(),
            params,
        );
        let high_bound = raw_bm25_block_range_upper_bound(
            *plan.terms[0].blocks.last().unwrap(),
            &plan.terms,
            stats.avg_doc_len(),
            params,
        );
        assert!(low_bound < high_bound);
        let seeded_floor = (low_bound + high_bound) / 2.0;
        let expected =
            retrieve_bm25_raw_segment_with_stats(&segment, &query, 10, params, &stats).unwrap();

        let unseeded = retrieve_bm25_raw_file_with_stats_pruned_blocks(
            &mut file_segment,
            &query,
            10,
            params,
            &stats,
            None,
        )
        .unwrap()
        .expect("fixture should enter the block-pruned path");
        let seeded = retrieve_bm25_raw_file_with_stats_pruned_blocks(
            &mut file_segment,
            &query,
            10,
            params,
            &stats,
            Some(seeded_floor),
        )
        .unwrap()
        .expect("fixture should enter the block-pruned path");

        assert_hits_close(&expected, &seeded.hits, "seeded block-pruned BM25");
        assert!(
            seeded.stats.term_blocks_pruned > unseeded.stats.term_blocks_pruned,
            "seeded floor should skip low-bound blocks before local top-k fills"
        );
    }

    #[test]
    fn raw_bm25_files_diagnostics_report_seeded_block_pruning() {
        const TERM: RawTermId = 10;
        const FILLER: RawTermId = 999;
        const TAIL_DOCS: u32 = 128;
        const SECOND_DOCS: u32 = 4096;

        let first: Vec<_> = (0..10)
            .map(|doc_id| (doc_id, vec![(TERM, 1_000), (FILLER, 9_000)]))
            .collect();
        let second: Vec<_> = (0..SECOND_DOCS)
            .map(|doc_offset| {
                let tf = if doc_offset >= SECOND_DOCS - TAIL_DOCS {
                    900
                } else {
                    1
                };
                (10_000 + doc_offset, vec![(TERM, tf)])
            })
            .collect();
        let mut index = InvertedIndex::new();
        for (doc_id, terms) in first.iter().chain(second.iter()) {
            add_raw_doc_to_memory_index(&mut index, *doc_id, terms);
        }

        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let query = vec![TERM];
        let params = Bm25Params::default();
        let stats = {
            let mut segments = [&mut first_segment, &mut second_segment];
            RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap()
        };

        let expected = raw_bm25_memory_hits(&index, &query, 10, params);
        let unseeded_pruned = retrieve_bm25_raw_file_with_search_stats(
            &mut first_segment,
            &query,
            10,
            params,
            &stats,
        )
        .unwrap()
        .stats
        .term_blocks_pruned
            + retrieve_bm25_raw_file_with_search_stats(
                &mut second_segment,
                &query,
                10,
                params,
                &stats,
            )
            .unwrap()
            .stats
            .term_blocks_pruned;

        let mut segments = [&mut first_segment, &mut second_segment];
        let result =
            retrieve_bm25_raw_files_with_diagnostics(&mut segments, &query, 10, params, &stats)
                .unwrap();

        assert_hits_close(&expected, &result.hits, "diagnostic multi-file BM25");
        assert_eq!(result.diagnostics.segments.segments_seen, 2);
        assert_eq!(result.diagnostics.segments.segments_scored, 2);
        assert_eq!(result.diagnostics.segments.segments_pruned, 0);
        assert!(
            result.diagnostics.term_blocks_pruned > unseeded_pruned,
            "multi-file diagnostics should expose extra seeded block pruning"
        );
        assert_eq!(
            result.diagnostics.term_blocks_seen,
            result.diagnostics.term_blocks_scored + result.diagnostics.term_blocks_pruned
        );
    }

    #[test]
    fn dense_bm25_range_uses_file_segment_doc_id_span() {
        let raw_docs = generated_raw_docs(128, 64, 8, 0xd35e_5eed, |i| 1_000_000 + i);
        let bytes = build_raw_bytes_with_doc_ids(&raw_docs);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        assert_eq!(
            dense_bm25_range(&mut file_segment, 128).unwrap(),
            Some((1_000_000, 128))
        );
    }

    #[test]
    fn raw_bm25_files_match_in_memory_index_with_global_stats() {
        let first = vec![
            (1, vec![(10, 3), (20, 1)]),
            (2, vec![(20, 5)]),
            (3, vec![(30, 1)]),
        ];
        let second = vec![
            (10, vec![(10, 1), (30, 3)]),
            (11, vec![(30, 2)]),
            (12, vec![(40, 4)]),
        ];
        let mut index = InvertedIndex::new();
        for (doc_id, terms) in first.iter().chain(second.iter()) {
            add_raw_doc_to_memory_index(&mut index, *doc_id, terms);
        }

        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let query = vec![10, 30, 10];
        let mut segments = [&mut first_segment, &mut second_segment];

        let stats = RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap();
        assert_eq!(stats.num_docs(), 6);
        assert_eq!(stats.df(10), Some(2));
        assert_eq!(stats.df(30), Some(3));
        assert_eq!(stats.df(40), None);

        let all_stats = RawBm25CorpusStats::from_raw_files_all_terms(&mut segments).unwrap();
        assert_eq!(all_stats.num_docs(), 6);
        assert_eq!(all_stats.df(10), Some(2));
        assert_eq!(all_stats.df(30), Some(3));
        assert_eq!(all_stats.df(40), Some(1));

        let raw_hits = retrieve_bm25_raw_files_with_stats(
            &mut segments,
            &query,
            10,
            Bm25Params::default(),
            &stats,
        )
        .unwrap();
        let all_stats_hits = retrieve_bm25_raw_files_with_stats(
            &mut segments,
            &query,
            10,
            Bm25Params::default(),
            &all_stats,
        )
        .unwrap();
        let auto_stats_hits =
            retrieve_bm25_raw_files(&mut segments, &query, 10, Bm25Params::default()).unwrap();
        assert_eq!(auto_stats_hits, raw_hits);
        assert_eq!(all_stats_hits, raw_hits);

        let memory_query: Vec<String> = query.iter().map(ToString::to_string).collect();
        let memory_hits = index
            .retrieve(&memory_query, 10, Bm25Params::default())
            .unwrap();

        assert_eq!(raw_hits.len(), memory_hits.len());
        for ((raw_doc, raw_score), (memory_doc, memory_score)) in
            raw_hits.iter().zip(memory_hits.iter())
        {
            assert_eq!(raw_doc, memory_doc);
            assert!(
                (raw_score - memory_score).abs() < 1e-6,
                "doc {raw_doc}: raw={raw_score}, memory={memory_score}"
            );
        }
    }

    #[test]
    fn raw_bm25_files_and_live_index_match_in_memory_index_with_global_stats() {
        let sealed = [
            (1, vec![(10, 3), (20, 1)]),
            (2, vec![(20, 5)]),
            (3, vec![(30, 1)]),
        ];
        let live = [
            (10, vec![(10, 1), (30, 7)]),
            (11, vec![(20, 2), (40, 10)]),
            (12, vec![(10, 4), (40, 1)]),
        ];
        let mut index = InvertedIndex::new();
        let mut live_index = PostingsIndex::new();
        for (doc_id, terms) in sealed.iter().chain(live.iter()) {
            add_raw_doc_to_memory_index(&mut index, *doc_id, terms);
            if *doc_id >= 10 {
                live_index.add_weighted_document(*doc_id, terms).unwrap();
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let sealed_path = dir.path().join("sealed.raw");
        std::fs::write(&sealed_path, build_raw_bytes_with_doc_ids(&sealed)).unwrap();
        let mut sealed_segment = RawSegmentFile::open(&sealed_path).unwrap();
        let query = vec![10, 30, 10, 40];
        let mut segments = [&mut sealed_segment];

        let stats =
            RawBm25CorpusStats::from_raw_files_and_index(&mut segments, &live_index, &query)
                .unwrap();
        assert_eq!(stats.num_docs(), 6);
        assert_eq!(stats.df(10), Some(3));
        assert_eq!(stats.df(30), Some(2));
        assert_eq!(stats.df(40), Some(2));

        let all_stats =
            RawBm25CorpusStats::from_raw_files_and_index_all_terms(&mut segments, &live_index)
                .unwrap();
        assert_eq!(all_stats.num_docs(), 6);
        assert_eq!(all_stats.df(10), Some(3));
        assert_eq!(all_stats.df(20), Some(3));
        assert_eq!(all_stats.df(30), Some(2));
        assert_eq!(all_stats.df(40), Some(2));
        assert_eq!(all_stats.df(99), None);

        let got = retrieve_bm25_raw_files_and_index_with_stats(
            &mut segments,
            &live_index,
            &query,
            10,
            Bm25Params::default(),
            &stats,
        )
        .unwrap();
        let auto_stats_got = retrieve_bm25_raw_files_and_index(
            &mut segments,
            &live_index,
            &query,
            10,
            Bm25Params::default(),
        )
        .unwrap();
        let all_stats_got = retrieve_bm25_raw_files_and_index_with_stats(
            &mut segments,
            &live_index,
            &query,
            10,
            Bm25Params::default(),
            &all_stats,
        )
        .unwrap();
        let expected = raw_bm25_memory_hits(&index, &query, 10, Bm25Params::default());

        assert_hits_close(&expected, &got, "raw files plus live shard BM25");
        assert_eq!(auto_stats_got, got);
        assert_eq!(all_stats_got, got);
    }

    #[test]
    fn raw_bm25_files_and_live_index_use_live_threshold_to_skip_low_bound_files() {
        const TERM: RawTermId = 7;

        let sealed = vec![(1, vec![(TERM, 1), (99, 99)])];
        let mut sealed_bytes = build_raw_bytes_with_doc_ids(&sealed);
        let dir = tempfile::tempdir().unwrap();
        let good_path = dir.path().join("sealed-good.raw");
        std::fs::write(&good_path, &sealed_bytes).unwrap();
        let good_segment = RawSegmentFile::open(&good_path).unwrap();
        let block = good_segment.posting_blocks(TERM).unwrap()[0];
        sealed_bytes[block.postings_offset() as usize] ^= 0xFF;

        let corrupted_path = dir.path().join("sealed-corrupted.raw");
        std::fs::write(&corrupted_path, sealed_bytes).unwrap();
        let mut corrupted_segment = RawSegmentFile::open(&corrupted_path).unwrap();

        let mut live_index = PostingsIndex::new();
        for doc_id in 100..110 {
            live_index
                .add_weighted_document(doc_id, &[(TERM, 20)])
                .unwrap();
        }

        let query = vec![TERM];
        let params = Bm25Params::default();
        let mut segments = [&mut corrupted_segment];
        let stats =
            RawBm25CorpusStats::from_raw_files_and_index(&mut segments, &live_index, &query)
                .unwrap();

        let hits = retrieve_bm25_raw_files_and_index_with_stats(
            &mut segments,
            &live_index,
            &query,
            10,
            params,
            &stats,
        )
        .unwrap();

        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|(doc_id, _)| *doc_id >= 100));
    }

    #[test]
    fn raw_bm25_files_do_not_prune_equal_bound_tie() {
        let first = vec![(10, vec![(7, 5)])];
        let second = vec![(1, vec![(7, 5)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];
        let query = vec![7];
        let stats = RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap();

        assert_eq!(
            retrieve_bm25_raw_files_with_stats(
                &mut segments,
                &query,
                1,
                Bm25Params::default(),
                &stats
            )
            .unwrap()
            .first()
            .map(|(doc_id, _)| *doc_id),
            Some(1)
        );
    }

    #[test]
    fn raw_bm25_files_prune_dominated_segment() {
        // Segment A holds a short, very-high-tf doc, so after it is scored the
        // top-k threshold sits above segment B's best-possible score. B's only
        // posting for the term has tf=1, so its per-segment upper bound
        // (raw.rs:365-383) falls below that threshold and the MaxScore skip at
        // raw.rs:390 fires. This is the pruning regime the k=10-over-6-docs and
        // equal-bound-tie tests never enter.
        const TERM: RawTermId = 7;
        let first = vec![(100, vec![(TERM, 20)])];
        let second = vec![(1, vec![(TERM, 1)])];

        let mut index = InvertedIndex::new();
        for (doc_id, terms) in first.iter().chain(second.iter()) {
            add_raw_doc_to_memory_index(&mut index, *doc_id, terms);
        }

        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];
        let query = vec![TERM];
        let params = Bm25Params::default();
        let stats = RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap();

        let raw_result =
            retrieve_bm25_raw_files_with_search_stats(&mut segments, &query, 1, params, &stats)
                .unwrap();

        // (a) The pruned result equals the brute-force in-memory oracle.
        let memory_query: Vec<String> = query.iter().map(ToString::to_string).collect();
        let memory_hits = index.retrieve(&memory_query, 1, params).unwrap();
        assert_eq!(raw_result.hits.len(), memory_hits.len());
        for ((raw_doc, raw_score), (memory_doc, memory_score)) in
            raw_result.hits.iter().zip(memory_hits.iter())
        {
            assert_eq!(raw_doc, memory_doc);
            assert!(
                (raw_score - memory_score).abs() < 1e-6,
                "doc {raw_doc}: raw={raw_score}, memory={memory_score}"
            );
        }
        assert_eq!(raw_result.hits.first().map(|(doc, _)| *doc), Some(100));
        assert_eq!(
            raw_result.stats,
            RawBm25SearchStats {
                segments_seen: 2,
                segments_scored: 1,
                segments_pruned: 1,
            }
        );

        // (b) The skip actually fired: recompute segment B's upper bound with
        // the same arithmetic as the pruning path and assert it is below the
        // realized top score that became the threshold after scoring segment A.
        // A parity check alone can pass without ever entering the pruning
        // regime, so this inequality is the load-bearing assertion.
        let avg_doc_len = stats.avg_doc_len();
        let corpus_df = stats.df(TERM).unwrap();
        let idf = bm25_idf_plus1(stats.num_docs(), corpus_df);
        let b_max_tf = segments[1].max_weight(TERM).unwrap();
        let b_upper_bound = idf * bm25_tf_upper_bound(b_max_tf as f32, avg_doc_len, params);
        let realized_top = raw_result.hits[0].1;
        assert!(
            b_upper_bound < realized_top,
            "segment B upper bound {b_upper_bound} must be below the realized top \
             score {realized_top} for the MaxScore skip to fire",
        );
    }

    #[test]
    fn raw_bm25_files_prune_disjoint_vocab_segments() {
        let dir = tempfile::tempdir().unwrap();
        let mut all_docs = Vec::new();
        let mut opened = Vec::new();

        for segment_index in 0..4 {
            let term_base = (segment_index * 100) as RawTermId;
            let doc_base = (segment_index * 10) as DocId;
            let docs = vec![
                (doc_base, vec![(term_base, 10), (term_base + 1, 1)]),
                (doc_base + 1, vec![(term_base, 5)]),
            ];
            all_docs.extend(docs.iter().cloned());

            let path = dir.path().join(format!("segment-{segment_index}.raw"));
            std::fs::write(&path, build_raw_bytes_with_doc_ids(&docs)).unwrap();
            opened.push(RawSegmentFile::open(&path).unwrap());
        }

        let index = build_memory_index_with_doc_ids(&all_docs);
        let params = Bm25Params::default();
        let query = vec![0, 1];
        let expected = raw_bm25_memory_hits(&index, &query, 2, params);

        let mut segments: Vec<_> = opened.iter_mut().collect();
        let stats = RawBm25CorpusStats::from_raw_files(&mut segments, &query).unwrap();
        let result =
            retrieve_bm25_raw_files_with_search_stats(&mut segments, &query, 2, params, &stats)
                .unwrap();

        assert_hits_close(&expected, &result.hits, "disjoint-vocab raw BM25");
        assert_eq!(
            result.stats,
            RawBm25SearchStats {
                segments_seen: 4,
                segments_scored: 1,
                segments_pruned: 3,
            }
        );
    }

    #[test]
    fn raw_bm25_files_require_stats_before_pruning_present_terms() {
        let first = vec![(10, vec![(7, 100)])];
        let second = vec![(1, vec![(8, 1)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let err = retrieve_bm25_raw_files_with_stats(
            &mut segments,
            &[7, 8],
            1,
            Bm25Params::default(),
            &RawBm25CorpusStats::new(2, 1.0, [(7, 1)]),
        )
        .unwrap_err();

        assert!(matches!(err, RawScoringError::MissingCorpusStats(8)));
    }

    #[test]
    fn raw_bm25_files_return_empty_for_absent_terms_without_stats() {
        let first = vec![(10, vec![(7, 5)])];
        let second = vec![(1, vec![(8, 5)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, build_raw_bytes_with_doc_ids(&first)).unwrap();
        std::fs::write(&second_path, build_raw_bytes_with_doc_ids(&second)).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let hits = retrieve_bm25_raw_files_with_stats(
            &mut segments,
            &[99],
            10,
            Bm25Params::default(),
            &RawBm25CorpusStats::new(2, 1.0, []),
        )
        .unwrap();

        assert!(hits.is_empty());
    }

    #[test]
    fn raw_bm25_file_with_stats_requires_stats_for_present_terms() {
        let raw_docs = build_raw_docs();
        let bytes = build_raw_bytes(&raw_docs);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let stats = RawBm25CorpusStats::new(4, 2.0, []);

        let err = retrieve_bm25_raw_file_with_stats(
            &mut file_segment,
            &[10],
            3,
            Bm25Params::default(),
            &stats,
        )
        .unwrap_err();

        assert!(matches!(err, RawScoringError::MissingCorpusStats(10)));
    }

    #[test]
    fn raw_bm25_preserves_duplicate_query_weight() {
        let raw_docs = build_raw_docs();
        let bytes = build_raw_bytes(&raw_docs);
        let segment = RawSegment::open(&bytes).unwrap();

        let single = retrieve_bm25_raw_segment(&segment, &[10], 10, Bm25Params::default()).unwrap();
        let duplicated =
            retrieve_bm25_raw_segment(&segment, &[10, 10], 10, Bm25Params::default()).unwrap();

        assert_eq!(single.len(), duplicated.len());
        for ((single_doc, single_score), (dup_doc, dup_score)) in
            single.iter().zip(duplicated.iter())
        {
            assert_eq!(single_doc, dup_doc);
            assert!((dup_score - single_score * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn raw_bm25_path_selection_ignores_absent_terms() {
        #[derive(Default)]
        struct PolicyReader {
            postings_calls: usize,
            streamed_calls: usize,
        }

        impl RawSegmentRead for PolicyReader {
            type Error = std::convert::Infallible;

            fn num_docs(&self) -> u32 {
                100
            }

            fn max_doc_id(&self) -> DocId {
                10
            }

            fn avg_doc_len(&self) -> f32 {
                1.0
            }

            fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error> {
                Ok((term_id == 7) as u32)
            }

            fn document_len(&mut self, _doc_id: DocId) -> Result<Option<u32>, Self::Error> {
                Ok(Some(1))
            }

            fn for_each_document_len(
                &mut self,
                _visit: impl FnMut(DocId, u32),
            ) -> Result<(), Self::Error> {
                Ok(())
            }

            fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
                self.postings_calls += 1;
                Ok((term_id == 7).then_some((3, 1)).into_iter().collect())
            }

            fn for_each_posting_with_document_len(
                &mut self,
                term_id: RawTermId,
                mut visit: impl FnMut(DocId, u32, u32),
            ) -> Result<(), Self::Error> {
                self.streamed_calls += 1;
                if term_id == 7 {
                    visit(3, 1, 1);
                }
                Ok(())
            }
        }

        let mut reader = PolicyReader::default();
        let mut query = vec![7];
        query.extend(1_000..1_064);
        let stats = RawBm25CorpusStats::new(100, 1.0, [(7, 1)]);

        let result = retrieve_bm25_raw_with_stats_and_search_stats(
            &mut reader,
            &query,
            1,
            Bm25Params::default(),
            &stats,
        )
        .unwrap();

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.stats.path, RawBm25FileSearchPath::SparseStream);
        assert_eq!(result.stats.terms_scored, 1);
        assert_eq!(result.stats.touched_postings_upper_bound, 1);
        assert_eq!(reader.streamed_calls, 1);
        assert_eq!(reader.postings_calls, 0);
    }

    #[test]
    fn raw_term_multiplicities_sort_and_count_duplicates() {
        let terms = raw_term_multiplicities(&[30, 10, 30, 20, 10, 10]);
        let observed: Vec<_> = terms
            .iter()
            .map(|term| (term.term_id, term.count, term.idf))
            .collect();

        assert_eq!(observed, vec![(10, 3, 0.0), (20, 1, 0.0), (30, 2, 0.0)]);
    }

    #[test]
    fn raw_bm25_streaming_policy_keeps_expanded_queries_cached() {
        assert!(should_stream_bm25(2, 20_000));
        assert!(should_stream_bm25(STREAMING_BM25_MAX_QUERY_TERMS, 20_000));
        assert!(!should_stream_bm25(
            STREAMING_BM25_MAX_QUERY_TERMS + 1,
            STREAMING_BM25_MIN_TOUCHED_POSTINGS - 1
        ));
        assert!(should_stream_bm25(
            STREAMING_BM25_MAX_QUERY_TERMS + 1,
            STREAMING_BM25_MIN_TOUCHED_POSTINGS
        ));
    }

    #[test]
    fn raw_bm25_block_pruning_requires_nonnegative_parameters() {
        assert!(can_prune_raw_bm25_blocks(Bm25Params::default()));
        assert!(can_prune_raw_bm25_blocks(Bm25Params::bm25l()));
        assert!(can_prune_raw_bm25_blocks(Bm25Params::bm25plus()));
        assert!(!can_prune_raw_bm25_blocks(Bm25Params {
            k1: 1.2,
            b: -0.25,
            variant: Bm25Variant::Standard,
        }));
        assert!(!can_prune_raw_bm25_blocks(Bm25Params {
            k1: 1.2,
            b: 0.75,
            variant: Bm25Variant::bm25plus_with_delta(-1.0),
        }));
    }

    #[test]
    fn dense_doc_length_prefill_requires_broad_queries() {
        assert!(!should_prefill_dense_doc_lengths(1, 0));
        assert!(should_prefill_dense_doc_lengths(1, 1));
        assert!(!should_prefill_dense_doc_lengths(20_000, 9_999));
        assert!(should_prefill_dense_doc_lengths(20_000, 10_000));
        assert!(should_prefill_dense_doc_lengths(20_000, 100_000));
    }

    #[test]
    fn bm25_tf_upper_bound_covers_supported_variants() {
        let params = [
            Bm25Params::default(),
            Bm25Params::bm25l(),
            Bm25Params {
                k1: 1.2,
                b: -3.0,
                variant: Bm25Variant::bm25l_with_delta(0.25),
            },
            Bm25Params::bm25plus(),
        ];

        for params in params {
            let bound = bm25_tf_upper_bound(7.0, 4.0, params);
            for tf in [1.0, 3.0, 7.0] {
                for doc_len in [0.0, 1.0, 4.0, 20.0] {
                    let score = bm25_tf_score(tf, doc_len, 4.0, params);
                    assert!(
                        score <= bound + 1e-6,
                        "score={score} exceeded bound={bound} for {params:?}"
                    );
                }
            }
        }
    }
}
