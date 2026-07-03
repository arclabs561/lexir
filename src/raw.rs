//! BM25 retrieval over `postings::raw` numeric segments.
//!
//! This is the out-of-core lexical path: callers keep their own term-id mapping
//! and use `postings` raw segments for storage, while `lexir` supplies BM25
//! scoring and deterministic top-k ranking.

use crate::bm25::{bm25_tf_score, Bm25Params};
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
    /// The underlying raw segment reader failed.
    Source(E),
}

impl<E: fmt::Display> fmt::Display for RawScoringError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => f.write_str("empty query"),
            Self::EmptyIndex => f.write_str("empty index"),
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
            Self::EmptyQuery | Self::EmptyIndex => None,
        }
    }
}

trait RawSegmentRead {
    type Error;

    fn num_docs(&self) -> u32;
    fn avg_doc_len(&self) -> f32;
    fn df(&mut self, term_id: RawTermId) -> Result<u32, Self::Error>;
    fn document_len(&mut self, doc_id: DocId) -> Result<Option<u32>, Self::Error>;
    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error>;
}

impl RawSegmentRead for RawSegment<'_> {
    type Error = postings::raw::Error;

    fn num_docs(&self) -> u32 {
        RawSegment::num_docs(self)
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

    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
        RawSegment::postings(self, term_id)?.collect()
    }
}

impl RawSegmentRead for RawSegmentFile {
    type Error = RawSegmentFileError;

    fn num_docs(&self) -> u32 {
        RawSegmentFile::num_docs(self)
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

    fn postings(&mut self, term_id: RawTermId) -> Result<Vec<(DocId, u32)>, Self::Error> {
        RawSegmentFile::postings(self, term_id)
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
    retrieve_bm25_raw(&mut segment, query_terms, k, params)
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
    retrieve_bm25_raw(segment, query_terms, k, params)
}

fn retrieve_bm25_raw<S>(
    segment: &mut S,
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
) -> Result<Vec<(DocId, f32)>, RawScoringError<S::Error>>
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
    if k == 0 {
        return Ok(Vec::new());
    }
    let avg_doc_len = segment.avg_doc_len();
    if avg_doc_len == 0.0 {
        return Ok(Vec::new());
    }

    let mut terms = raw_term_multiplicities(query_terms);
    let mut touched_upper_bound = 0usize;
    for term in &mut terms {
        let df = segment.df(term.term_id).map_err(RawScoringError::Source)?;
        term.idf = bm25_idf_plus1(num_docs, df);
        if term.idf != 0.0 {
            touched_upper_bound = touched_upper_bound.saturating_add(df as usize);
        }
    }

    let mut scores = HashMap::with_capacity(touched_upper_bound.min(num_docs as usize));
    let mut doc_lengths = HashMap::with_capacity(touched_upper_bound.min(num_docs as usize));

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

    Ok(top_k_positive_scored_docs(scores, k))
}

#[derive(Debug)]
struct WeightedRawTerm {
    term_id: RawTermId,
    count: usize,
    idf: f32,
}

fn raw_term_multiplicities(query_terms: &[RawTermId]) -> Vec<WeightedRawTerm> {
    let mut counts: Vec<WeightedRawTerm> = Vec::with_capacity(query_terms.len());

    for &term_id in query_terms {
        if let Some(seen) = counts.iter_mut().find(|seen| seen.term_id == term_id) {
            seen.count += 1;
        } else {
            counts.push(WeightedRawTerm {
                term_id,
                count: 1,
                idf: 0.0,
            });
        }
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
            let mut expanded = Vec::new();
            for &(term_id, weight) in terms {
                for _ in 0..weight {
                    expanded.push(term_id.to_string());
                }
            }
            index.add_document(doc_id as DocId, &expanded);
        }
        index
    }

    fn build_raw_bytes(raw_docs: &[Vec<(RawTermId, u32)>]) -> Vec<u8> {
        let docs: Vec<_> = raw_docs
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        write_u64_u32_segment(&docs).unwrap()
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
}
