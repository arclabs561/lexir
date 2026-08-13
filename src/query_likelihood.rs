//! Query likelihood language model retrieval.
//!
//! Ranks documents by \(P(Q|D)\): the probability that a document language model generated the
//! query. This is a foundational probabilistic retrieval approach (Ponte & Croft, 1998).
//!
//! This implementation is **index-only** (no raw document text required): it uses the same
//! postings-backed corpus statistics as BM25/TF-IDF.

use crate::bm25::InvertedIndex;
use crate::query_terms::term_multiplicities;
use crate::ranking::top_k_non_nan_scored_docs;
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
        if p_smoothed == 0.0 {
            return f32::NEG_INFINITY;
        }
        log_score += count as f32 * p_smoothed.ln();
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
        if p_smoothed == 0.0 {
            return f32::NEG_INFINITY;
        }
        log_score += count as f32 * p_smoothed.ln();
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

    let terms = term_multiplicities(query_terms);

    // Smoothing gives non-matching documents probability mass, so exact retrieval must score
    // every live document. A postings-only candidate set is not exact for either model.
    let (corpus_term_freqs, corpus_size) = index.corpus_stats_cached();
    let corpus_term_freqs = corpus_term_freqs.as_ref();
    let scored_documents = index.document_ids().map(|doc_id| {
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
    Ok(top_k_non_nan_scored_docs(scored_documents, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::InvertedIndex;
    use std::collections::BTreeMap;

    fn oracle(
        docs: &[(u32, Vec<&str>)],
        query: &[String],
        k: usize,
        smoothing: SmoothingMethod,
    ) -> Vec<(u32, f32)> {
        let corpus_size = docs.iter().map(|(_, terms)| terms.len()).sum::<usize>() as f32;
        let mut corpus_tf = BTreeMap::<&str, usize>::new();
        for term in docs.iter().flat_map(|(_, terms)| terms) {
            *corpus_tf.entry(term).or_default() += 1;
        }
        let mut query_counts = BTreeMap::<&str, usize>::new();
        for term in query {
            *query_counts.entry(term).or_default() += 1;
        }

        let mut scores: Vec<_> = docs
            .iter()
            .map(|(doc_id, terms)| {
                let mut score = 0.0_f32;
                for (term, count) in &query_counts {
                    let tf = terms.iter().filter(|candidate| *candidate == term).count() as f32;
                    let p_corpus = *corpus_tf.get(term).unwrap_or(&0) as f32 / corpus_size;
                    let probability = match smoothing {
                        SmoothingMethod::JelinekMercer { lambda } => {
                            let lambda = lambda.clamp(0.0, 1.0);
                            let p_doc = if terms.is_empty() {
                                0.0
                            } else {
                                tf / terms.len() as f32
                            };
                            lambda * p_doc + (1.0 - lambda) * p_corpus
                        }
                        SmoothingMethod::Dirichlet { mu } => {
                            let mu = mu.max(0.0);
                            let denominator = terms.len() as f32 + mu;
                            if denominator > 0.0 {
                                (tf + mu * p_corpus) / denominator
                            } else {
                                0.0
                            }
                        }
                    };
                    if probability == 0.0 {
                        score = f32::NEG_INFINITY;
                        break;
                    }
                    score += *count as f32 * probability.ln();
                }
                (*doc_id, score)
            })
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scores.truncate(k);
        scores
    }

    #[test]
    fn query_likelihood_matches_exhaustive_oracle() {
        let docs = vec![
            (7, vec!["a", "a", "b"]),
            (2, vec!["b", "c"]),
            (9, vec!["a", "c"]),
            (4, vec!["c", "c"]),
        ];
        let mut index = InvertedIndex::new();
        for (doc_id, terms) in &docs {
            index.add_document(
                *doc_id,
                &terms.iter().map(ToString::to_string).collect::<Vec<_>>(),
            );
        }

        let queries = [vec!["a"], vec!["a", "a"], vec!["a", "b"], vec!["missing"]];
        let methods = [
            SmoothingMethod::JelinekMercer { lambda: 0.0 },
            SmoothingMethod::JelinekMercer { lambda: 0.4 },
            SmoothingMethod::JelinekMercer { lambda: 1.0 },
            SmoothingMethod::Dirichlet { mu: 0.0 },
            SmoothingMethod::Dirichlet { mu: 3.0 },
        ];

        for query in queries {
            let query = query
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            for smoothing in methods {
                for k in 0..=docs.len() + 1 {
                    let actual = retrieve_query_likelihood(
                        &index,
                        &query,
                        k,
                        QueryLikelihoodParams { smoothing },
                    )
                    .unwrap();
                    let expected = oracle(&docs, &query, k, smoothing);
                    assert_eq!(actual.len(), expected.len());
                    for ((actual_doc, actual_score), (expected_doc, expected_score)) in
                        actual.iter().zip(&expected)
                    {
                        assert_eq!(
                            actual_doc, expected_doc,
                            "query={query:?}, smoothing={smoothing:?}, k={k}"
                        );
                        if expected_score.is_infinite() {
                            assert_eq!(actual_score, expected_score);
                        } else {
                            assert!((actual_score - expected_score).abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }

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
