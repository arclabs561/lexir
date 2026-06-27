//! Closed-form BM25 reference test.
//!
//! Invariant: the BM25 score for a query Q on a document D equals the value
//! computed by hand from the BM25 definition, under params k1 = 1.2, b = 0.75.
//!
//! The oracle below is derived from the BM25 definition (Robertson & Zaragoza
//! 2009) plus the "+1" IDF variant documented in `rankfns::bm25_idf_plus1`. It
//! is NOT obtained by running the code and pasting its output. The hand
//! arithmetic is shown in the comments and the final numeric constants are
//! independently confirmed (Wolfram Alpha: doc0 + doc1 = 1.56304689...).
//!
//! Formulas (rankfns 0.1.3, the version lexir resolves):
//!   idf(t)              = ln( ((N - df + 0.5) / (df + 0.5)) + 1 )
//!   tf_norm(tf, len)    = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len/avgdl))
//!   score(D, Q)         = sum over t in Q of idf(t) * tf_norm(tf_{t,D}, len_D)

use lexir::bm25::{Bm25Params, InvertedIndex};

/// Build a 4-doc corpus with hand-chosen lengths so avgdl is exactly 3.0.
///
/// Query term: "apple".
///   doc 0: [apple, apple, banana]            len 3,  tf(apple) = 2
///   doc 1: [apple, cherry, date, fig]        len 4,  tf(apple) = 1
///   doc 2: [banana, cherry]                  len 2,  tf(apple) = 0
///   doc 3: [grape, kiwi, lemon]              len 3,  tf(apple) = 0
///
/// Total length = 3 + 4 + 2 + 3 = 12, N = 4, avgdl = 12 / 4 = 3.0.
/// df(apple) = 2 (docs 0 and 1).
fn build_corpus() -> InvertedIndex {
    let mut ix = InvertedIndex::new();
    ix.add_document(0, &["apple".into(), "apple".into(), "banana".into()]);
    ix.add_document(
        1,
        &["apple".into(), "cherry".into(), "date".into(), "fig".into()],
    );
    ix.add_document(2, &["banana".into(), "cherry".into()]);
    ix.add_document(3, &["grape".into(), "kiwi".into(), "lemon".into()]);
    ix
}

#[test]
fn bm25_score_matches_hand_computed_reference() {
    let ix = build_corpus();

    // Pin the params the hand computation assumes. If the crate defaults ever
    // change, this fails loudly rather than silently invalidating the oracle.
    let params = Bm25Params::default();
    assert_eq!(params.k1, 1.2, "oracle assumes k1 = 1.2");
    assert_eq!(params.b, 0.75, "oracle assumes b = 0.75");

    // Assert every input the oracle depends on, so a divergence in how postings
    // counts tokens / df / avgdl fails here instead of corrupting the score.
    assert_eq!(ix.num_docs(), 4);
    assert_eq!(ix.avg_doc_len(), 3.0, "avgdl = 12 tokens / 4 docs");
    assert_eq!(ix.document_length(0), 3);
    assert_eq!(ix.document_length(1), 4);
    assert_eq!(ix.term_frequency(0, "apple"), 2);
    assert_eq!(ix.term_frequency(1, "apple"), 1);
    assert_eq!(ix.doc_frequency("apple"), 2);

    // idf(apple) = ln( ((4 - 2 + 0.5) / (2 + 0.5)) + 1 )
    //            = ln( (2.5 / 2.5) + 1 ) = ln(1 + 1) = ln(2) = 0.6931471805599453
    let idf = std::f64::consts::LN_2;
    assert!(
        (ix.idf("apple") as f64 - idf).abs() < 1e-6,
        "idf(apple) should be ln(2), got {}",
        ix.idf("apple")
    );

    let query = vec!["apple".into()];

    // doc 0: tf = 2, len = 3, avgdl = 3.0
    //   tf_norm = (2 * (1.2 + 1)) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3/3))
    //           = 4.4 / (2 + 1.2 * 1.0) = 4.4 / 3.2 = 1.375
    //   score   = ln(2) * 1.375 = 0.9530773732699247
    let oracle_doc0 = 0.9530773732699247_f64;
    let got0 = ix.score(0, &query, params) as f64;
    assert!(
        (got0 - oracle_doc0).abs() < 1e-5,
        "doc 0 BM25 score: expected {oracle_doc0}, got {got0}"
    );

    // doc 1: tf = 1, len = 4, avgdl = 3.0
    //   tf_norm = (1 * 2.2) / (1 + 1.2 * (1 - 0.75 + 0.75 * 4/3))
    //           = 2.2 / (1 + 1.2 * 1.25) = 2.2 / 2.5 = 0.88
    //   score   = ln(2) * 0.88 = 0.6099695188927519
    let oracle_doc1 = 0.6099695188927519_f64;
    let got1 = ix.score(1, &query, params) as f64;
    assert!(
        (got1 - oracle_doc1).abs() < 1e-5,
        "doc 1 BM25 score: expected {oracle_doc1}, got {got1}"
    );

    // docs 2 and 3 do not contain "apple": score must be exactly 0.
    assert_eq!(ix.score(2, &query, params), 0.0);
    assert_eq!(ix.score(3, &query, params), 0.0);
}

#[test]
fn bm25_ranking_is_unambiguous() {
    // doc 0 has both higher term frequency (2 vs 1) AND a shorter document
    // (3 vs 4 tokens, i.e. less length penalty) than doc 1, so its BM25 score is
    // unambiguously higher under any k1 >= 0, b in [0, 1]. The float epsilon
    // assertions above are paired here with an order check that does not depend
    // on exact values.
    let ix = build_corpus();
    let hits = ix
        .retrieve(&["apple".into()], 10, Bm25Params::default())
        .unwrap();

    // Only docs 0 and 1 contain "apple"; 2 and 3 are excluded (score 0).
    assert_eq!(hits.len(), 2, "only two docs contain the query term");
    assert_eq!(hits[0].0, 0, "doc 0 must rank first");
    assert_eq!(hits[1].0, 1, "doc 1 must rank second");
    assert!(
        hits[0].1 > hits[1].1,
        "doc 0 score {} must exceed doc 1 score {}",
        hits[0].1,
        hits[1].1
    );
}
