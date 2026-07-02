use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lexir::bm25::{Bm25Params, InvertedIndex};
use lexir::query_likelihood::{retrieve_query_likelihood, QueryLikelihoodParams};
use lexir::tfidf::{retrieve_tfidf, TfIdfParams};

const N_DOCS: u32 = 20_000;
const VOCAB_SIZE: usize = 5_000;
const TERMS_PER_DOC: usize = 80;

fn zipf_sample(rng: &mut u64, vocab_size: usize) -> usize {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let u = (*rng as f64) / (u64::MAX as f64);
    let n = vocab_size as f64;
    let k = (u * (n + 1.0_f64).ln()).exp() - 1.0;
    (k as usize).min(vocab_size - 1)
}

fn term_str(id: usize) -> String {
    format!("t{id:05}")
}

fn build_index() -> InvertedIndex {
    let mut index = InvertedIndex::new();
    let mut rng = 0xdeadbeef_cafebabe_u64;

    for doc_id in 0..N_DOCS {
        let terms: Vec<String> = (0..TERMS_PER_DOC)
            .map(|_| term_str(zipf_sample(&mut rng, VOCAB_SIZE)))
            .collect();
        index.add_document(doc_id, &terms);
    }

    index
}

fn query_terms(index: &InvertedIndex, count: usize, min_df: u32) -> Vec<String> {
    (0..VOCAB_SIZE)
        .map(term_str)
        .filter(|term| index.doc_frequency(term) >= min_df)
        .take(count)
        .collect()
}

fn bench_bm25_retrieve(c: &mut Criterion) {
    let index = build_index();
    let params = Bm25Params::default();
    let mut group = c.benchmark_group("bm25_retrieve");

    for n in [2usize, 8] {
        let query = query_terms(&index, n, 20);
        group.bench_with_input(BenchmarkId::new("terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(
                    index
                        .retrieve(black_box(query.as_slice()), 10, params)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_tfidf_retrieve(c: &mut Criterion) {
    let index = build_index();
    let params = TfIdfParams::default();
    let mut group = c.benchmark_group("tfidf_retrieve");

    for n in [2usize, 8] {
        let query = query_terms(&index, n, 20);
        group.bench_with_input(BenchmarkId::new("terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(
                    retrieve_tfidf(black_box(&index), black_box(query.as_slice()), 10, params)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_query_likelihood_retrieve(c: &mut Criterion) {
    let index = build_index();
    let params = QueryLikelihoodParams::default();
    let mut group = c.benchmark_group("query_likelihood_retrieve");

    for n in [2usize, 8] {
        let query = query_terms(&index, n, 20);
        group.bench_with_input(BenchmarkId::new("terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(
                    retrieve_query_likelihood(
                        black_box(&index),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bm25_retrieve,
    bench_tfidf_retrieve,
    bench_query_likelihood_retrieve
);
criterion_main!(benches);
