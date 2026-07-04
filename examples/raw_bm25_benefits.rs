//! Measure raw BM25 pruning benefits with deterministic fixtures.
//!
//! Run:
//! `cargo run --release --example raw_bm25_benefits --features raw-segment`

use std::hint::black_box;
use std::time::{Duration, Instant};

use lexir::bm25::Bm25Params;
use lexir::raw::{
    retrieve_bm25_raw_file_with_search_stats, retrieve_bm25_raw_files_and_index_with_diagnostics,
    retrieve_bm25_raw_files_with_diagnostics, RawBm25CorpusStats, RawBm25FileSearchStats,
    RawBm25SearchDiagnostics,
};
use postings::raw::{
    write_u64_u32_segment_sorted_from_iter_to, RawDocument, RawSegmentFile, RawTermId,
};
use postings::PostingsIndex;

const SAMPLES: usize = 7;
const ITERS: usize = 200;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "case\tmode\tmedian_ns\tsegments_seen\tsegments_scored\tsegments_pruned\tblocks_seen\tblocks_scored\tblocks_pruned\thits"
    );
    segment_pruning()?;
    seeded_block_pruning()?;
    live_seeded_segment_pruning()?;
    Ok(())
}

fn segment_pruning() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, mut segments, stats, query) = build_segment_pruning_fixture()?;
    let params = Bm25Params::default();
    let mut refs: Vec<_> = segments.iter_mut().collect();
    let pruned =
        retrieve_bm25_raw_files_with_diagnostics(refs.as_mut_slice(), &query, 10, params, &stats)?;
    drop(refs);
    let pruned_time = median_per_iter(|| {
        let mut refs: Vec<_> = segments.iter_mut().collect();
        retrieve_bm25_raw_files_with_diagnostics(refs.as_mut_slice(), &query, 10, params, &stats)
            .unwrap()
    });

    let forced = retrieve_all_files_with_diagnostics(&mut segments, &query, 10, params, &stats)?;
    let forced_time = median_per_iter(|| {
        retrieve_all_files_with_diagnostics(&mut segments, &query, 10, params, &stats).unwrap()
    });

    print_row(
        "segment_pruning",
        "pruned",
        pruned_time,
        &pruned.diagnostics,
        pruned.hits.len(),
    );
    print_row(
        "segment_pruning",
        "forced_files",
        forced_time,
        &forced.diagnostics,
        forced.hits.len(),
    );
    Ok(())
}

fn seeded_block_pruning() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, mut segments, stats, query) = build_seeded_block_fixture()?;
    let params = Bm25Params::default();
    let mut refs: Vec<_> = segments.iter_mut().collect();
    let seeded =
        retrieve_bm25_raw_files_with_diagnostics(refs.as_mut_slice(), &query, 10, params, &stats)?;
    drop(refs);
    let seeded_time = median_per_iter(|| {
        let mut refs: Vec<_> = segments.iter_mut().collect();
        retrieve_bm25_raw_files_with_diagnostics(refs.as_mut_slice(), &query, 10, params, &stats)
            .unwrap()
    });

    let unseeded = retrieve_all_files_with_diagnostics(&mut segments, &query, 10, params, &stats)?;
    let unseeded_time = median_per_iter(|| {
        retrieve_all_files_with_diagnostics(&mut segments, &query, 10, params, &stats).unwrap()
    });

    print_row(
        "seeded_block_pruning",
        "seeded",
        seeded_time,
        &seeded.diagnostics,
        seeded.hits.len(),
    );
    print_row(
        "seeded_block_pruning",
        "unseeded_files",
        unseeded_time,
        &unseeded.diagnostics,
        unseeded.hits.len(),
    );
    Ok(())
}

fn live_seeded_segment_pruning() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, mut segments, live_index, stats, query) = build_live_seeded_fixture()?;
    let params = Bm25Params::default();
    let mut refs: Vec<_> = segments.iter_mut().collect();
    let result = retrieve_bm25_raw_files_and_index_with_diagnostics(
        refs.as_mut_slice(),
        &live_index,
        &query,
        10,
        params,
        &stats,
    )?;
    drop(refs);
    let elapsed = median_per_iter(|| {
        let mut refs: Vec<_> = segments.iter_mut().collect();
        retrieve_bm25_raw_files_and_index_with_diagnostics(
            refs.as_mut_slice(),
            &live_index,
            &query,
            10,
            params,
            &stats,
        )
        .unwrap()
    });

    print_row(
        "live_seeded_segment_pruning",
        "live_seeded",
        elapsed,
        &result.diagnostics,
        result.hits.len(),
    );
    Ok(())
}

fn median_per_iter<T>(mut run: impl FnMut() -> T) -> Duration {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for _ in 0..ITERS {
            black_box(run());
        }
        samples.push(start.elapsed() / ITERS as u32);
    }
    samples.sort_unstable();
    samples[SAMPLES / 2]
}

fn print_row(
    case: &str,
    mode: &str,
    elapsed: Duration,
    diagnostics: &RawBm25SearchDiagnostics,
    hits: usize,
) {
    println!(
        "{case}\t{mode}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{hits}",
        elapsed.as_nanos(),
        diagnostics.segments.segments_seen,
        diagnostics.segments.segments_scored,
        diagnostics.segments.segments_pruned,
        diagnostics.term_blocks_seen,
        diagnostics.term_blocks_scored,
        diagnostics.term_blocks_pruned,
    );
}

struct DiagnosticHits {
    hits: Vec<(u32, f32)>,
    diagnostics: RawBm25SearchDiagnostics,
}

fn retrieve_all_files_with_diagnostics(
    segments: &mut [RawSegmentFile],
    query: &[RawTermId],
    k: usize,
    params: Bm25Params,
    stats: &RawBm25CorpusStats,
) -> Result<DiagnosticHits, Box<dyn std::error::Error>> {
    let mut candidates = Vec::new();
    let mut diagnostics = RawBm25SearchDiagnostics::default();
    diagnostics.segments.segments_seen = segments.len();

    for segment in segments {
        let result = retrieve_bm25_raw_file_with_search_stats(segment, query, k, params, stats)?;
        diagnostics.segments.segments_scored += 1;
        add_file_stats(&mut diagnostics, result.stats);
        candidates.extend(result.hits);
    }

    Ok(DiagnosticHits {
        hits: top_k(candidates, k),
        diagnostics,
    })
}

fn add_file_stats(diagnostics: &mut RawBm25SearchDiagnostics, stats: RawBm25FileSearchStats) {
    diagnostics.terms_scored += stats.terms_scored;
    diagnostics.touched_postings_upper_bound += stats.touched_postings_upper_bound;
    diagnostics.dense_slots += stats.dense_slots;
    diagnostics.term_blocks_seen += stats.term_blocks_seen;
    diagnostics.term_blocks_scored += stats.term_blocks_scored;
    diagnostics.term_blocks_pruned += stats.term_blocks_pruned;
}

fn top_k(mut hits: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    hits.retain(|(_, score)| score.is_finite() && *score > 0.0);
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    hits.truncate(k);
    hits
}

type FilesFixture = (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    RawBm25CorpusStats,
    Vec<RawTermId>,
);

fn build_segment_pruning_fixture() -> Result<FilesFixture, Box<dyn std::error::Error>> {
    const QUERY_TERM: RawTermId = 7;
    const COLD_SEGMENTS: usize = 16;
    const COLD_DOCS_PER_SEGMENT: usize = 8_192;

    let dir = tempfile::tempdir()?;
    let query = vec![QUERY_TERM];
    let hot_docs = vec![vec![(QUERY_TERM, 20), (999, 80)]; 128];
    let mut segments = vec![write_raw_file(&dir, "hot.raw", &hot_docs, 0)?];

    for segment_id in 0..COLD_SEGMENTS {
        let cold_docs = vec![vec![(QUERY_TERM, 1), (999, 99)]; COLD_DOCS_PER_SEGMENT];
        segments.push(write_raw_file(
            &dir,
            &format!("cold-{segment_id}.raw"),
            &cold_docs,
            10_000 + (segment_id * COLD_DOCS_PER_SEGMENT) as u32,
        )?);
    }

    let mut refs: Vec<_> = segments.iter_mut().collect();
    let stats = RawBm25CorpusStats::from_raw_files(refs.as_mut_slice(), &query)?;
    drop(refs);
    Ok((dir, segments, stats, query))
}

fn build_seeded_block_fixture() -> Result<FilesFixture, Box<dyn std::error::Error>> {
    const QUERY_TERM: RawTermId = 7;
    const COLD_DOCS: u32 = 4096;
    const HOT_TAIL_DOCS: u32 = 128;

    let dir = tempfile::tempdir()?;
    let query = vec![QUERY_TERM];
    let first_docs = vec![vec![(QUERY_TERM, 1_000), (999, 9_000)]; 10];
    let second_docs: Vec<_> = (0..COLD_DOCS)
        .map(|doc_id| {
            if doc_id >= COLD_DOCS - HOT_TAIL_DOCS {
                vec![(QUERY_TERM, 900)]
            } else {
                vec![(QUERY_TERM, 1)]
            }
        })
        .collect();
    let mut segments = vec![
        write_raw_file(&dir, "seeded-block-seed.raw", &first_docs, 0)?,
        write_raw_file(&dir, "seeded-block-tail.raw", &second_docs, 10_000)?,
    ];
    let mut refs: Vec<_> = segments.iter_mut().collect();
    let stats = RawBm25CorpusStats::from_raw_files(refs.as_mut_slice(), &query)?;
    drop(refs);
    Ok((dir, segments, stats, query))
}

type LiveFixture = (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    PostingsIndex<RawTermId, u32>,
    RawBm25CorpusStats,
    Vec<RawTermId>,
);

fn build_live_seeded_fixture() -> Result<LiveFixture, Box<dyn std::error::Error>> {
    const QUERY_TERM: RawTermId = 7;
    const COLD_SEGMENTS: usize = 64;
    const COLD_DOCS_PER_SEGMENT: usize = 64;
    const LIVE_DOCS: usize = 10;

    let dir = tempfile::tempdir()?;
    let query = vec![QUERY_TERM];
    let mut segments = Vec::new();
    for segment_id in 0..COLD_SEGMENTS {
        let cold_docs = vec![vec![(QUERY_TERM, 1), (999, 99)]; COLD_DOCS_PER_SEGMENT];
        segments.push(write_raw_file(
            &dir,
            &format!("live-prune-cold-{segment_id}.raw"),
            &cold_docs,
            (segment_id * COLD_DOCS_PER_SEGMENT) as u32,
        )?);
    }

    let live_docs = vec![vec![(QUERY_TERM, 20)]; LIVE_DOCS];
    let live_index = build_raw_live_index(&live_docs, 1_000_000)?;
    let mut refs: Vec<_> = segments.iter_mut().collect();
    let stats =
        RawBm25CorpusStats::from_raw_files_and_index(refs.as_mut_slice(), &live_index, &query)?;
    drop(refs);
    Ok((dir, segments, live_index, stats, query))
}

fn write_raw_file(
    dir: &tempfile::TempDir,
    name: &str,
    raw_docs: &[Vec<(RawTermId, u32)>],
    start_doc_id: u32,
) -> Result<RawSegmentFile, Box<dyn std::error::Error>> {
    let docs = raw_docs
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(start_doc_id + doc_id as u32, terms));
    let path = dir.path().join(name);
    let mut file = std::fs::File::create(&path)?;
    write_u64_u32_segment_sorted_from_iter_to(docs, &mut file)?;
    drop(file);
    Ok(RawSegmentFile::open(path)?)
}

fn build_raw_live_index(
    raw_docs: &[Vec<(RawTermId, u32)>],
    start_doc_id: u32,
) -> Result<PostingsIndex<RawTermId, u32>, Box<dyn std::error::Error>> {
    let mut index = PostingsIndex::new();
    for (doc_offset, terms) in raw_docs.iter().enumerate() {
        index.add_weighted_document(start_doc_id + doc_offset as u32, terms)?;
    }
    Ok(index)
}
