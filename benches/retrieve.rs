use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lexir::bm25::{Bm25Params, InvertedIndex};
use lexir::query_likelihood::{retrieve_query_likelihood, QueryLikelihoodParams};
use lexir::tfidf::{retrieve_tfidf, TfIdfParams};
#[cfg(feature = "raw-segment")]
use postings::raw::{
    write_u64_u32_segment_sorted_from_iter_to, RawDocument, RawSegmentFile, RawTermId,
};
#[cfg(feature = "raw-segment")]
use postings::PostingsIndex;

const N_DOCS: u32 = 20_000;
const VOCAB_SIZE: usize = 5_000;
const TERMS_PER_DOC: usize = 80;
#[cfg(feature = "raw-segment")]
const PARTITIONED_RAW_SEGMENTS: usize = 64;
#[cfg(feature = "raw-segment")]
const PARTITIONED_RAW_DOCS_PER_SEGMENT: usize = 128;
#[cfg(feature = "raw-segment")]
const PARTITIONED_RAW_VOCAB_PER_SEGMENT: usize = 512;
#[cfg(feature = "raw-segment")]
const PARTITIONED_RAW_TERMS_PER_DOC: usize = 32;
#[cfg(feature = "raw-segment")]
const PARTITIONED_RAW_QUERY_TERMS: usize = 8;

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

fn duplicate_query_terms(index: &InvertedIndex, unique: usize, repeat: usize) -> Vec<String> {
    let mut terms = Vec::with_capacity(unique * repeat);
    for term in query_terms(index, unique, 20) {
        for _ in 0..repeat {
            terms.push(term.clone());
        }
    }
    terms
}

#[cfg(feature = "raw-segment")]
fn build_raw_docs() -> Vec<Vec<(RawTermId, u32)>> {
    let mut docs = Vec::with_capacity(N_DOCS as usize);
    let mut rng = 0xdeadbeef_cafebabe_u64;

    for _ in 0..N_DOCS {
        let mut terms = Vec::with_capacity(TERMS_PER_DOC);
        for _ in 0..TERMS_PER_DOC {
            terms.push((zipf_sample(&mut rng, VOCAB_SIZE) as RawTermId, 1));
        }
        docs.push(terms);
    }

    docs
}

#[cfg(feature = "raw-segment")]
fn raw_query_terms(segment: &RawSegmentFile, count: usize, min_df: u32) -> Vec<RawTermId> {
    (0..VOCAB_SIZE as RawTermId)
        .filter(|&term| segment.df(term).unwrap() >= min_df)
        .take(count)
        .collect()
}

#[cfg(feature = "raw-segment")]
fn write_raw_file(
    dir: &tempfile::TempDir,
    name: &str,
    raw_docs: &[Vec<(RawTermId, u32)>],
    start_doc_id: u32,
) -> RawSegmentFile {
    let docs = raw_docs
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(start_doc_id + doc_id as u32, terms));
    let path = dir.path().join(name);
    let mut file = std::fs::File::create(&path).unwrap();
    write_u64_u32_segment_sorted_from_iter_to(docs, &mut file).unwrap();
    drop(file);
    RawSegmentFile::open(path).unwrap()
}

#[cfg(feature = "raw-segment")]
fn build_raw_live_index(
    raw_docs: &[Vec<(RawTermId, u32)>],
    start_doc_id: u32,
) -> PostingsIndex<RawTermId, u32> {
    let mut index = PostingsIndex::new();
    for (doc_offset, terms) in raw_docs.iter().enumerate() {
        index
            .add_weighted_document(start_doc_id + doc_offset as u32, terms)
            .unwrap();
    }
    index
}

#[cfg(feature = "raw-segment")]
fn build_prunable_raw_files() -> (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    lexir::raw::RawBm25CorpusStats,
    Vec<RawTermId>,
) {
    const QUERY_TERM: RawTermId = 7;
    const COLD_SEGMENTS: usize = 16;
    const COLD_DOCS_PER_SEGMENT: usize = 8_192;

    let dir = tempfile::tempdir().unwrap();
    let query = vec![QUERY_TERM];
    let hot_docs = vec![vec![(QUERY_TERM, 20), (999, 80)]; 128];
    let mut segments = vec![write_raw_file(&dir, "hot.raw", &hot_docs, 0)];

    for segment_id in 0..COLD_SEGMENTS {
        let cold_docs = vec![vec![(QUERY_TERM, 1), (999, 99)]; COLD_DOCS_PER_SEGMENT];
        segments.push(write_raw_file(
            &dir,
            &format!("cold-{segment_id}.raw"),
            &cold_docs,
            10_000 + (segment_id * COLD_DOCS_PER_SEGMENT) as u32,
        ));
    }

    let mut segment_refs: Vec<_> = segments.iter_mut().collect();
    let stats = lexir::raw::RawBm25CorpusStats::from_raw_files(&mut segment_refs, &query).unwrap();
    drop(segment_refs);

    (dir, segments, stats, query)
}

#[cfg(feature = "raw-segment")]
fn build_live_prunable_raw_files() -> (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    PostingsIndex<RawTermId, u32>,
    lexir::raw::RawBm25CorpusStats,
    Vec<RawTermId>,
) {
    const QUERY_TERM: RawTermId = 7;
    const COLD_SEGMENTS: usize = 64;
    const COLD_DOCS_PER_SEGMENT: usize = 64;
    const LIVE_DOCS: usize = 10;

    let dir = tempfile::tempdir().unwrap();
    let query = vec![QUERY_TERM];
    let mut segments = Vec::new();
    for segment_id in 0..COLD_SEGMENTS {
        let cold_docs = vec![vec![(QUERY_TERM, 1), (999, 99)]; COLD_DOCS_PER_SEGMENT];
        segments.push(write_raw_file(
            &dir,
            &format!("live-prune-cold-{segment_id}.raw"),
            &cold_docs,
            (segment_id * COLD_DOCS_PER_SEGMENT) as u32,
        ));
    }

    let live_docs = vec![vec![(QUERY_TERM, 20)]; LIVE_DOCS];
    let live_index = build_raw_live_index(&live_docs, 1_000_000);
    let mut segment_refs: Vec<_> = segments.iter_mut().collect();
    let stats = lexir::raw::RawBm25CorpusStats::from_raw_files_and_index(
        &mut segment_refs,
        &live_index,
        &query,
    )
    .unwrap();
    drop(segment_refs);

    (dir, segments, live_index, stats, query)
}

#[cfg(feature = "raw-segment")]
fn build_partitioned_raw_files() -> (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    lexir::raw::RawBm25CorpusStats,
    Vec<RawTermId>,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut segments = Vec::new();

    for segment_id in 0..PARTITIONED_RAW_SEGMENTS {
        let term_base = (segment_id * PARTITIONED_RAW_VOCAB_PER_SEGMENT) as RawTermId;
        let doc_base = (segment_id * PARTITIONED_RAW_DOCS_PER_SEGMENT) as u32;
        let mut docs = Vec::with_capacity(PARTITIONED_RAW_DOCS_PER_SEGMENT);

        for doc_offset in 0..PARTITIONED_RAW_DOCS_PER_SEGMENT {
            let mut terms = std::collections::BTreeMap::new();
            for term_offset in 0..PARTITIONED_RAW_TERMS_PER_DOC {
                let local_term =
                    (doc_offset * 37 + term_offset * 17) % PARTITIONED_RAW_VOCAB_PER_SEGMENT;
                let tf = 1 + ((doc_offset + term_offset + segment_id) % 7) as u32;
                *terms
                    .entry(term_base + local_term as RawTermId)
                    .or_insert(0) += tf;
            }
            docs.push(terms.into_iter().collect());
        }

        segments.push(write_raw_file(
            &dir,
            &format!("partitioned-{segment_id}.raw"),
            &docs,
            doc_base,
        ));
    }

    let query: Vec<_> = (0..PARTITIONED_RAW_QUERY_TERMS)
        .map(|term_offset| (term_offset * 17 % PARTITIONED_RAW_VOCAB_PER_SEGMENT) as RawTermId)
        .collect();
    let mut segment_refs: Vec<_> = segments.iter_mut().collect();
    let stats = lexir::raw::RawBm25CorpusStats::from_raw_files(&mut segment_refs, &query).unwrap();
    drop(segment_refs);

    (dir, segments, stats, query)
}

#[cfg(feature = "raw-segment")]
fn top_k_bench_docs(mut docs: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    docs.retain(|(_, score)| score.is_finite() && *score > 0.0);
    docs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    docs.truncate(k);
    docs
}

#[cfg(feature = "raw-segment")]
fn retrieve_bm25_raw_files_without_segment_pruning(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &lexir::raw::RawBm25CorpusStats,
) -> Vec<(u32, f32)> {
    let mut candidates = Vec::with_capacity(k.saturating_mul(segments.len()));
    for segment in segments {
        candidates.extend(
            lexir::raw::retrieve_bm25_raw_file_with_stats(segment, query_terms, k, params, stats)
                .unwrap(),
        );
    }
    top_k_bench_docs(candidates, k)
}

fn bench_bm25_retrieve(c: &mut Criterion) {
    let index = build_index();
    let params = Bm25Params::default();
    let mut group = c.benchmark_group("bm25_retrieve");

    for n in [1usize, 2, 8, 32] {
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

    let duplicate_query = duplicate_query_terms(&index, 2, 4);
    group.bench_with_input(
        BenchmarkId::new("duplicate_terms", duplicate_query.len()),
        &duplicate_query,
        |b, query| {
            b.iter(|| {
                black_box(
                    index
                        .retrieve(black_box(query.as_slice()), 10, params)
                        .unwrap(),
                );
            });
        },
    );

    group.finish();
}

#[cfg(feature = "raw-segment")]
fn bench_raw_bm25_retrieve(c: &mut Criterion) {
    let raw_docs = build_raw_docs();
    let docs = raw_docs
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms));
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut writer = std::fs::File::create(file.path()).unwrap();
    write_u64_u32_segment_sorted_from_iter_to(docs, &mut writer).unwrap();
    drop(writer);
    let mut segment = RawSegmentFile::open(file.path()).unwrap();
    let params = Bm25Params::default();
    let mut group = c.benchmark_group("raw_bm25_retrieve");

    for n in [1usize, 2, 8, 32] {
        let query = raw_query_terms(&segment, n, 20);
        group.bench_with_input(BenchmarkId::new("file_terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_file(
                        black_box(&mut segment),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        });
    }

    let diagnostic_query = raw_query_terms(&segment, 8, 20);
    let diagnostic_stats = {
        let mut segments = [&mut segment];
        lexir::raw::RawBm25CorpusStats::from_raw_files(&mut segments, &diagnostic_query).unwrap()
    };
    group.bench_with_input(
        BenchmarkId::new("file_terms_search_stats", diagnostic_query.len()),
        &diagnostic_query,
        |b, query| {
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_file_with_search_stats(
                        black_box(&mut segment),
                        black_box(query.as_slice()),
                        10,
                        params,
                        black_box(&diagnostic_stats),
                    )
                    .unwrap(),
                );
            });
        },
    );

    let duplicate_query: Vec<_> = raw_query_terms(&segment, 32, 20)
        .into_iter()
        .flat_map(|term| std::iter::repeat(term).take(16))
        .collect();
    group.bench_with_input(
        BenchmarkId::new("file_duplicate_terms", duplicate_query.len()),
        &duplicate_query,
        |b, query| {
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_file(
                        black_box(&mut segment),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        },
    );

    let mut one_present_many_absent_query = raw_query_terms(&segment, 1, 20);
    one_present_many_absent_query.extend((VOCAB_SIZE as RawTermId)..(VOCAB_SIZE as RawTermId + 64));
    group.bench_with_input(
        BenchmarkId::new(
            "file_one_present_many_absent",
            one_present_many_absent_query.len(),
        ),
        &one_present_many_absent_query,
        |b, query| {
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_file(
                        black_box(&mut segment),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        },
    );

    let dir = tempfile::tempdir().unwrap();
    let chunk_len = raw_docs.len().div_ceil(4);
    let mut multi_segments: Vec<_> = raw_docs
        .chunks(chunk_len)
        .enumerate()
        .map(|(chunk_id, chunk)| {
            write_raw_file(
                &dir,
                &format!("chunk-{chunk_id}.raw"),
                chunk,
                (chunk_id * chunk_len) as u32,
            )
        })
        .collect();
    let chunk_len_64 = raw_docs.len().div_ceil(64);
    let mut multi_segments_64: Vec<_> = raw_docs
        .chunks(chunk_len_64)
        .enumerate()
        .map(|(chunk_id, chunk)| {
            write_raw_file(
                &dir,
                &format!("chunk-64-{chunk_id}.raw"),
                chunk,
                (chunk_id * chunk_len_64) as u32,
            )
        })
        .collect();
    let multi_query = raw_query_terms(&segment, 8, 20);
    group.bench_with_input(
        BenchmarkId::new("files_terms", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = multi_segments.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files(
                        black_box(segments.as_mut_slice()),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("files_query_stats_build", multi_query.len()),
        &multi_query,
        |b, query| {
            b.iter(|| {
                let mut segments: Vec<_> = multi_segments.iter_mut().collect();
                black_box(
                    lexir::raw::RawBm25CorpusStats::from_raw_files(
                        black_box(segments.as_mut_slice()),
                        black_box(query.as_slice()),
                    )
                    .unwrap(),
                );
            });
        },
    );

    group.bench_function("files_all_terms_stats_build", |b| {
        b.iter(|| {
            let mut segments: Vec<_> = multi_segments.iter_mut().collect();
            black_box(
                lexir::raw::RawBm25CorpusStats::from_raw_files_all_terms(black_box(
                    segments.as_mut_slice(),
                ))
                .unwrap(),
            );
        });
    });

    let mut all_stats_refs: Vec<_> = multi_segments.iter_mut().collect();
    let all_term_stats =
        lexir::raw::RawBm25CorpusStats::from_raw_files_all_terms(&mut all_stats_refs).unwrap();
    drop(all_stats_refs);
    let mut all_stats_refs_64: Vec<_> = multi_segments_64.iter_mut().collect();
    let all_term_stats_64 =
        lexir::raw::RawBm25CorpusStats::from_raw_files_all_terms(&mut all_stats_refs_64).unwrap();
    drop(all_stats_refs_64);
    group.bench_with_input(
        BenchmarkId::new("files_terms_all_stats", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = multi_segments.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files_with_stats(
                        black_box(segments.as_mut_slice()),
                        black_box(query.as_slice()),
                        10,
                        params,
                        black_box(&all_term_stats),
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("files_terms_all_stats_64", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = multi_segments_64.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files_with_stats(
                        black_box(segments.as_mut_slice()),
                        black_box(query.as_slice()),
                        10,
                        params,
                        black_box(&all_term_stats_64),
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("files_terms_all_stats_64_search_stats", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = multi_segments_64.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files_with_search_stats(
                        black_box(segments.as_mut_slice()),
                        black_box(query.as_slice()),
                        10,
                        params,
                        black_box(&all_term_stats_64),
                    )
                    .unwrap(),
                );
            });
        },
    );

    let live_doc_count = raw_docs.len() / 10;
    let live_start = raw_docs.len() - live_doc_count;
    let live_index = build_raw_live_index(&raw_docs[live_start..], live_start as u32);
    let mixed_chunk_len = raw_docs[..live_start].len().div_ceil(4);
    let mut mixed_segments: Vec<_> = raw_docs[..live_start]
        .chunks(mixed_chunk_len)
        .enumerate()
        .map(|(chunk_id, chunk)| {
            write_raw_file(
                &dir,
                &format!("mixed-chunk-{chunk_id}.raw"),
                chunk,
                (chunk_id * mixed_chunk_len) as u32,
            )
        })
        .collect();
    let mut mixed_stats_refs: Vec<_> = mixed_segments.iter_mut().collect();
    let mixed_stats = lexir::raw::RawBm25CorpusStats::from_raw_files_and_index(
        &mut mixed_stats_refs,
        &live_index,
        &multi_query,
    )
    .unwrap();
    drop(mixed_stats_refs);
    group.bench_with_input(
        BenchmarkId::new("files_and_live_index_terms", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = mixed_segments.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files_and_index(
                        black_box(segments.as_mut_slice()),
                        black_box(&live_index),
                        black_box(query.as_slice()),
                        10,
                        params,
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("files_and_live_index_stats_build", multi_query.len()),
        &multi_query,
        |b, query| {
            b.iter(|| {
                let mut segments: Vec<_> = mixed_segments.iter_mut().collect();
                black_box(
                    lexir::raw::RawBm25CorpusStats::from_raw_files_and_index(
                        black_box(segments.as_mut_slice()),
                        black_box(&live_index),
                        black_box(query.as_slice()),
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.bench_function("files_and_live_index_all_terms_stats_build", |b| {
        b.iter(|| {
            let mut segments: Vec<_> = mixed_segments.iter_mut().collect();
            black_box(
                lexir::raw::RawBm25CorpusStats::from_raw_files_and_index_all_terms(
                    black_box(segments.as_mut_slice()),
                    black_box(&live_index),
                )
                .unwrap(),
            );
        });
    });
    group.bench_with_input(
        BenchmarkId::new("files_and_live_index_terms_with_stats", multi_query.len()),
        &multi_query,
        |b, query| {
            let mut segments: Vec<_> = mixed_segments.iter_mut().collect();
            b.iter(|| {
                black_box(
                    lexir::raw::retrieve_bm25_raw_files_and_index_with_stats(
                        black_box(segments.as_mut_slice()),
                        black_box(&live_index),
                        black_box(query.as_slice()),
                        10,
                        params,
                        black_box(&mixed_stats),
                    )
                    .unwrap(),
                );
            });
        },
    );
    let (
        _live_prune_dir,
        mut live_prune_segments,
        live_prune_index,
        live_prune_stats,
        live_prune_query,
    ) = build_live_prunable_raw_files();
    group.bench_function("files_and_live_index_live_prunes_64", |b| {
        let mut segments: Vec<_> = live_prune_segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                lexir::raw::retrieve_bm25_raw_files_and_index_with_stats(
                    black_box(segments.as_mut_slice()),
                    black_box(&live_prune_index),
                    black_box(live_prune_query.as_slice()),
                    10,
                    params,
                    black_box(&live_prune_stats),
                )
                .unwrap(),
            );
        });
    });

    let (_prunable_dir, mut prunable_segments, prunable_stats, prunable_query) =
        build_prunable_raw_files();
    group.bench_function("files_prunable_with_stats", |b| {
        let mut segments: Vec<_> = prunable_segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                lexir::raw::retrieve_bm25_raw_files_with_stats(
                    black_box(segments.as_mut_slice()),
                    black_box(prunable_query.as_slice()),
                    10,
                    params,
                    black_box(&prunable_stats),
                )
                .unwrap(),
            );
        });
    });

    let (_exact_dir, mut exact_segments, exact_stats, exact_query) = build_prunable_raw_files();
    group.bench_function("files_forced_all_segments_with_stats", |b| {
        let mut segments: Vec<_> = exact_segments.iter_mut().collect();
        b.iter(|| {
            black_box(retrieve_bm25_raw_files_without_segment_pruning(
                black_box(segments.as_mut_slice()),
                black_box(exact_query.as_slice()),
                10,
                params,
                black_box(&exact_stats),
            ));
        });
    });

    let (_partitioned_dir, mut partitioned_segments, partitioned_stats, partitioned_query) =
        build_partitioned_raw_files();
    {
        let mut segments: Vec<_> = partitioned_segments.iter_mut().collect();
        let result = lexir::raw::retrieve_bm25_raw_files_with_search_stats(
            &mut segments,
            &partitioned_query,
            10,
            params,
            &partitioned_stats,
        )
        .unwrap();
        assert_eq!(result.stats.segments_seen, PARTITIONED_RAW_SEGMENTS);
        assert_eq!(result.stats.segments_scored, 1);
        assert_eq!(result.stats.segments_pruned, PARTITIONED_RAW_SEGMENTS - 1);
    }
    group.bench_function("files_partitioned_search_stats_64", |b| {
        let mut segments: Vec<_> = partitioned_segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                lexir::raw::retrieve_bm25_raw_files_with_search_stats(
                    black_box(segments.as_mut_slice()),
                    black_box(partitioned_query.as_slice()),
                    10,
                    params,
                    black_box(&partitioned_stats),
                )
                .unwrap(),
            );
        });
    });

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

    let duplicate_query = duplicate_query_terms(&index, 2, 4);
    group.bench_with_input(
        BenchmarkId::new("duplicate_terms", duplicate_query.len()),
        &duplicate_query,
        |b, query| {
            b.iter(|| {
                black_box(
                    retrieve_tfidf(black_box(&index), black_box(query.as_slice()), 10, params)
                        .unwrap(),
                );
            });
        },
    );

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

    let duplicate_query = duplicate_query_terms(&index, 2, 4);
    group.bench_with_input(
        BenchmarkId::new("duplicate_terms", duplicate_query.len()),
        &duplicate_query,
        |b, query| {
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
        },
    );

    group.finish();
}

#[cfg(feature = "raw-segment")]
criterion_group!(
    benches,
    bench_bm25_retrieve,
    bench_raw_bm25_retrieve,
    bench_tfidf_retrieve,
    bench_query_likelihood_retrieve
);

#[cfg(not(feature = "raw-segment"))]
criterion_group!(
    benches,
    bench_bm25_retrieve,
    bench_tfidf_retrieve,
    bench_query_likelihood_retrieve
);
criterion_main!(benches);
