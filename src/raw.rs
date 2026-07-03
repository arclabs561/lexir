//! BM25 retrieval over `postings::raw` numeric segments.
//!
//! This is the out-of-core lexical path: callers keep their own term-id mapping
//! and use `postings` raw segments for storage, while `lexir` supplies BM25
//! scoring and deterministic top-k ranking.

use crate::bm25::{bm25_tf_score, Bm25Params, Bm25Variant};
use crate::ranking::top_k_positive_scored_docs;
use postings::raw::{RawSegment, RawSegmentFile, RawSegmentFileError, RawTermId};
use postings::DocId;
use rankfns::bm25_idf_plus1;
use std::collections::HashMap;
use std::fmt;

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

/// Hits and segment-level diagnostics from a multi-file raw BM25 search.
#[derive(Clone, Debug, PartialEq)]
pub struct RawBm25SearchResult {
    /// Top-k hits sorted by descending BM25 score, then document id.
    pub hits: Vec<(DocId, f32)>,
    /// Segment-pruning diagnostics for the search.
    pub stats: RawBm25SearchStats,
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
    retrieve_bm25_raw_with_stats(segment, query_terms, k, params, stats)
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
    retrieve_bm25_raw_files_with_search_stats(segments, query_terms, k, params, stats)
        .map(|result| result.hits)
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
    if query_terms.is_empty() {
        return Err(RawScoringError::EmptyQuery);
    }
    if stats.num_docs() == 0 {
        return Err(RawScoringError::EmptyIndex);
    }
    let mut search_stats = RawBm25SearchStats {
        segments_seen: segments.len(),
        ..RawBm25SearchStats::default()
    };
    if k == 0 {
        return Ok(RawBm25SearchResult {
            hits: Vec::new(),
            stats: search_stats,
        });
    }

    let avg_doc_len = stats.avg_doc_len();
    if avg_doc_len <= 0.0 || !avg_doc_len.is_finite() {
        return Ok(RawBm25SearchResult {
            hits: Vec::new(),
            stats: search_stats,
        });
    }

    let terms = raw_term_multiplicities(query_terms);
    let mut candidates = Vec::with_capacity(k.saturating_mul(segments.len()));
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
            search_stats.segments_pruned += 1;
        }
    }
    order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut threshold = 0.0;
    for (index, upper_bound) in order {
        if candidates.len() >= k && upper_bound < threshold {
            search_stats.segments_pruned += 1;
            continue;
        }
        search_stats.segments_scored += 1;
        candidates.extend(retrieve_bm25_raw_file_with_stats(
            segments[index],
            query_terms,
            k,
            params,
            stats,
        )?);
        if candidates.len() >= k {
            candidates = top_k_positive_scored_docs(candidates, k);
            threshold = candidates.last().map_or(0.0, |(_, score)| *score);
        }
    }

    Ok(RawBm25SearchResult {
        hits: top_k_positive_scored_docs(candidates, k),
        stats: search_stats,
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
    if query_terms.is_empty() {
        return Err(RawScoringError::EmptyQuery);
    }
    if stats.num_docs() == 0 {
        return Err(RawScoringError::EmptyIndex);
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    let avg_doc_len = stats.avg_doc_len();
    if avg_doc_len <= 0.0 || !avg_doc_len.is_finite() {
        return Ok(Vec::new());
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

    if let Some(slots) = dense_bm25_slots(segment.max_doc_id(), touched_upper_bound) {
        let dense_plan = DenseRawBm25Plan {
            stream_postings: should_stream_bm25(terms.len(), touched_upper_bound),
            slots,
            prefill_doc_lengths: should_prefill_dense_doc_lengths(
                segment.num_docs(),
                touched_upper_bound,
            ),
        };
        return retrieve_bm25_raw_dense(segment, dense_plan, terms, k, params, avg_doc_len);
    }

    let stream_postings = should_stream_bm25(terms.len(), touched_upper_bound);
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

    Ok(top_k_positive_scored_docs(scores, k))
}

struct DenseRawBm25Plan {
    stream_postings: bool,
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
                    let slot = doc_id as usize;
                    if slot >= slots {
                        return;
                    }
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(doc_id);
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
                    let slot = doc_id as usize;
                    if slot < slots {
                        doc_lengths[slot] = doc_length;
                    }
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
                let slot = doc_id as usize;
                if slot >= slots {
                    continue;
                }
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(doc_id);
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
            .map(|doc_id| (doc_id, scores[doc_id as usize])),
        k,
    ))
}

fn should_prefill_dense_doc_lengths(num_docs: u32, touched_upper_bound: usize) -> bool {
    touched_upper_bound > 0 && touched_upper_bound >= (num_docs as usize).div_ceil(2)
}

const DENSE_BM25_MAX_SLOTS: usize = 1_000_000;
const STREAMING_BM25_MAX_QUERY_TERMS: usize = 8;
const STREAMING_BM25_MIN_TOUCHED_POSTINGS: usize = 1_000_000;

fn dense_bm25_slots(max_doc_id: DocId, touched_upper_bound: usize) -> Option<usize> {
    let slots = usize::try_from(max_doc_id).ok()?.checked_add(1)?;
    let useful_limit = touched_upper_bound.saturating_mul(2).max(4096);
    (slots <= DENSE_BM25_MAX_SLOTS && slots <= useful_limit).then_some(slots)
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

        let raw_hits = retrieve_bm25_raw_files_with_stats(
            &mut segments,
            &query,
            10,
            Bm25Params::default(),
            &stats,
        )
        .unwrap();
        let auto_stats_hits =
            retrieve_bm25_raw_files(&mut segments, &query, 10, Bm25Params::default()).unwrap();
        assert_eq!(auto_stats_hits, raw_hits);

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
