//! Fuzzy term expansion for lexical retrieval.
//!
//! This module integrates:
//! - `lexir` (BM25/TF‑IDF scoring),
//! - `postings` (candidate generation over docs),
//! - `gramdex` (candidate generation over *terms* via k‑grams).
//!
//! Design goal: keep this **optional** and **deterministic**.

use crate::bm25::InvertedIndex;
use crate::Error;

use gramdex::{char_kgrams, GramDex, PlannerConfig};
use std::collections::HashSet;

/// Configuration for fuzzy query term expansion.
#[derive(Debug, Clone)]
pub struct FuzzyConfig {
    /// Character k-gram size (Unicode-scalar grams).
    pub k: usize,
    /// Candidate planning/bailout thresholds (keeps expansion bounded).
    pub planner: PlannerConfig,
    /// Keep at most this many expansions per original query term.
    pub max_expansions_per_term: usize,
    /// Minimum Jaccard similarity between query-term grams and candidate-term grams.
    pub min_jaccard: f32,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            k: 3,
            planner: PlannerConfig::default(),
            max_expansions_per_term: 8,
            min_jaccard: 0.25,
        }
    }
}

/// A reusable “vocabulary sketch” for fuzzy expansion.
///
/// Build this once from an index’s vocabulary and reuse across queries.
#[derive(Debug)]
pub struct FuzzyVocab {
    cfg_k: usize,
    term_by_id: Vec<String>,
    grams_by_id: Vec<Vec<String>>, // sorted unique grams
    ix: GramDex,
}

impl FuzzyVocab {
    /// Build a fuzzy vocabulary index from the current `lexir` term set.
    pub fn from_index_terms(index: &InvertedIndex, k: usize) -> Result<Self, gramdex::Error> {
        // Ensure determinism: stable term ordering.
        let mut terms: Vec<String> = index.terms().map(|t| t.to_string()).collect();
        terms.sort();
        terms.dedup();

        let mut ix = GramDex::new();
        let mut grams_by_id: Vec<Vec<String>> = Vec::with_capacity(terms.len());
        for (id, term) in terms.iter().enumerate() {
            let mut grams = char_kgrams(term, k)?;
            grams.sort();
            grams.dedup();
            ix.add_document(id as u32, &grams);
            grams_by_id.push(grams);
        }

        Ok(Self {
            cfg_k: k,
            term_by_id: terms,
            grams_by_id,
            ix,
        })
    }

    fn term_grams(&self, term: &str) -> Result<Vec<String>, gramdex::Error> {
        let mut grams = char_kgrams(term, self.cfg_k)?;
        grams.sort();
        grams.dedup();
        Ok(grams)
    }

    /// Return candidate vocabulary terms for a query term, ordered by similarity desc then term asc.
    pub fn expand_term(
        &self,
        query_term: &str,
        cfg: &FuzzyConfig,
    ) -> Result<Vec<String>, gramdex::Error> {
        let q_grams = self.term_grams(query_term)?;
        if q_grams.is_empty() {
            return Ok(Vec::new());
        }

        let cand_ids = self.ix.candidates_union_bounded(&q_grams, cfg.planner);
        let mut scored: Vec<(f32, String)> = Vec::new();

        for id in cand_ids {
            let id_usize = id as usize;
            if id_usize >= self.term_by_id.len() {
                continue;
            }
            let term = &self.term_by_id[id_usize];
            if term == query_term {
                continue;
            }
            let j = jaccard_sorted(&q_grams, &self.grams_by_id[id_usize]);
            if j >= cfg.min_jaccard && j.is_finite() {
                scored.push((j, term.clone()));
            }
        }

        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(cfg.max_expansions_per_term);
        Ok(scored.into_iter().map(|(_, t)| t).collect())
    }
}

/// Expand a query term list using fuzzy vocabulary matching.
///
/// Policy:
/// - Always keep the original query terms.
/// - Only expand terms that are OOV (df == 0) in the `InvertedIndex`.
/// - Deduplicate and return a stable, deterministic ordering.
pub fn expand_query_terms(
    index: &InvertedIndex,
    vocab: &FuzzyVocab,
    query_terms: &[String],
    cfg: &FuzzyConfig,
) -> Result<Vec<String>, Error> {
    if query_terms.is_empty() {
        return Err(Error::EmptyQuery);
    }

    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Keep original terms in order.
    for t in query_terms {
        if seen.insert(t.clone()) {
            out.push(t.clone());
        }
    }

    // Expand only OOV terms.
    for t in query_terms {
        if index.doc_frequency(t) > 0 {
            continue;
        }
        if cfg.k == 0 {
            return Err(Error::InvalidFuzzyConfig("k must be >= 1"));
        }
        let ex = vocab
            .expand_term(t, cfg)
            .map_err(|_| Error::InvalidFuzzyConfig("k must be >= 1"))?;
        for cand in ex {
            if seen.insert(cand.clone()) {
                out.push(cand);
            }
        }
    }

    Ok(out)
}

fn jaccard_sorted(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0u32;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = (a.len() + b.len()) as u32 - inter;
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::Bm25Params;

    #[test]
    fn fuzzy_expansion_can_recover_near_misspell() {
        let mut ix = InvertedIndex::new();
        ix.add_document(1, &["color".to_string(), "theory".to_string()]);
        ix.add_document(2, &["colour".to_string(), "theatre".to_string()]);

        let vocab = FuzzyVocab::from_index_terms(&ix, 3).unwrap();
        let cfg = FuzzyConfig {
            k: 3,
            max_expansions_per_term: 8,
            min_jaccard: 0.2,
            ..Default::default()
        };

        // Query uses the “other spelling”; we should expand to include the indexed token too.
        let q = vec!["colour".to_string(), "theory".to_string()];
        let expanded = expand_query_terms(&ix, &vocab, &q, &cfg).unwrap();
        assert!(
            expanded.contains(&"color".to_string()) || expanded.contains(&"colour".to_string())
        );

        // And retrieval should not error.
        let _ = ix.retrieve(&expanded, 5, Bm25Params::default()).unwrap();
    }
}
