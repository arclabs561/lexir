#![cfg(feature = "fuzzy")]

//! Consumer-level parity for fuzzy vocabulary expansion followed by BM25 retrieval.
//!
//! The oracle deliberately retains the source documents and scans them directly;
//! it does not inspect `lexir`, `postings`, or `gramdex` internals.

use std::collections::{BTreeMap, HashSet};

use lexir::bm25::{Bm25Params, InvertedIndex};
use lexir::fuzzy::{expand_query_terms, FuzzyConfig, FuzzyVocab};

type Documents = BTreeMap<u32, Vec<String>>;

fn terms(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn build_index(documents: &Documents) -> InvertedIndex {
    let mut index = InvertedIndex::new();
    for (&doc_id, document) in documents {
        index.add_document(doc_id, document);
    }
    index
}

fn fuzzy_retrieve(index: &InvertedIndex, query: &[&str], limit: usize) -> Vec<(u32, f32)> {
    let config = FuzzyConfig {
        k: 2,
        min_jaccard: 0.2,
        max_expansions_per_term: 8,
        ..FuzzyConfig::default()
    };
    let vocabulary = FuzzyVocab::from_index_terms(index, config.k).unwrap();
    let expanded = expand_query_terms(index, &vocabulary, &terms(query), &config).unwrap();
    index
        .retrieve(&expanded, limit, Bm25Params::default())
        .unwrap()
}

fn full_scan_bm25(documents: &Documents, query: &[String], limit: usize) -> Vec<(u32, f32)> {
    if documents.is_empty() || limit == 0 {
        return Vec::new();
    }

    let document_count = documents.len() as f32;
    let average_length = documents
        .values()
        .map(|document| document.len() as f32)
        .sum::<f32>()
        / document_count;
    let mut scored = Vec::new();

    for (&doc_id, document) in documents {
        let mut score = 0.0;
        for query_term in query {
            let document_frequency = documents
                .values()
                .filter(|candidate| candidate.iter().any(|term| term == query_term))
                .count() as f32;
            if document_frequency == 0.0 {
                continue;
            }

            let term_frequency = document.iter().filter(|term| *term == query_term).count() as f32;
            if term_frequency == 0.0 {
                continue;
            }

            let idf = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            let length_ratio = document.len() as f32 / average_length;
            let denominator = term_frequency + 1.2 * (1.0 - 0.75 + 0.75 * length_ratio);
            score += idf * (term_frequency * 2.2) / denominator;
        }
        if score > 0.0 {
            scored.push((doc_id, score));
        }
    }

    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
}

fn assert_full_scan_parity(
    documents: &Documents,
    index: &InvertedIndex,
    query: &[&str],
    limit: usize,
) {
    let config = FuzzyConfig {
        k: 2,
        min_jaccard: 0.2,
        max_expansions_per_term: 8,
        ..FuzzyConfig::default()
    };
    let vocabulary = FuzzyVocab::from_index_terms(index, config.k).unwrap();
    let expanded = expand_query_terms(index, &vocabulary, &terms(query), &config).unwrap();
    let actual = index
        .retrieve(&expanded, limit, Bm25Params::default())
        .unwrap();
    let expected = full_scan_bm25(documents, &expanded, limit);

    assert_eq!(
        actual.iter().map(|hit| hit.0).collect::<Vec<_>>(),
        expected.iter().map(|hit| hit.0).collect::<Vec<_>>(),
        "document ranking diverged for {query:?}, expanded as {expanded:?}"
    );
    for ((actual_id, actual_score), (expected_id, expected_score)) in actual.iter().zip(&expected) {
        assert_eq!(actual_id, expected_id);
        assert!(
            (actual_score - expected_score).abs() <= 1e-6,
            "score for document {actual_id} diverged: {actual_score} vs {expected_score}"
        );
    }
}

#[test]
fn fuzzy_pipeline_matches_source_document_full_scan() {
    let documents = BTreeMap::from([
        (2, terms(&["café", "café", "résumé"])),
        (5, terms(&["cafe", "resume", "resume"])),
        (7, terms(&["color", "theory"])),
        (11, terms(&["anchor", "tie"])),
        (19, terms(&["anchor", "tie"])),
    ]);
    let index = build_index(&documents);

    // Exact in-vocabulary terms are retained without fuzzy expansion.
    let config = FuzzyConfig {
        k: 2,
        min_jaccard: 0.2,
        ..FuzzyConfig::default()
    };
    let vocabulary = FuzzyVocab::from_index_terms(&index, config.k).unwrap();
    assert_eq!(
        expand_query_terms(&index, &vocabulary, &terms(&["color"]), &config).unwrap(),
        terms(&["color"])
    );

    assert_full_scan_parity(&documents, &index, &["color"], 10);
    assert_full_scan_parity(&documents, &index, &["colr"], 10);
    assert_full_scan_parity(&documents, &index, &["cafè"], 10);
    assert_full_scan_parity(&documents, &index, &["café", "café"], 10);
    assert_full_scan_parity(&documents, &index, &["zz"], 10);

    let tied = fuzzy_retrieve(&index, &["anchor"], 10);
    assert_eq!(tied.iter().map(|hit| hit.0).collect::<Vec<_>>(), [11, 19]);
    assert_eq!(tied[0].1, tied[1].1);

    let unicode_hits: HashSet<u32> = fuzzy_retrieve(&index, &["cafè"], 10)
        .into_iter()
        .map(|hit| hit.0)
        .collect();
    assert_eq!(unicode_hits, HashSet::from([2, 5]));
}

#[test]
fn rebuilt_fuzzy_pipeline_tracks_updates_and_deletes() {
    let mut documents = BTreeMap::from([
        (3, terms(&["planet", "orbit"])),
        (8, terms(&["planet", "orbit"])),
        (13, terms(&["comet", "tail"])),
    ]);
    let mut index = build_index(&documents);

    assert_full_scan_parity(&documents, &index, &["planett"], 10);

    let replacement = terms(&["galaxy", "galaxy", "cluster"]);
    documents.insert(3, replacement.clone());
    index.add_document(3, &replacement);
    assert_full_scan_parity(&documents, &index, &["galaxi"], 10);
    assert_full_scan_parity(&documents, &index, &["planet"], 10);

    documents.remove(&8);
    assert!(index.delete_document(8));
    assert!(!index.delete_document(8));
    assert_full_scan_parity(&documents, &index, &["planet"], 10);
    assert!(fuzzy_retrieve(&index, &["planett"], 10).is_empty());
}
