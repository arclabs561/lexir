//! Stream lexical documents into raw BM25 segment files plus one live shard.
//!
//! Run:
//! `cargo run --example raw_bm25_generation --features raw-segment`

use std::io::Write;

use lexir::bm25::Bm25Params;
use lexir::raw::{
    retrieve_bm25_raw_files_and_index_with_diagnostics, RawBm25CorpusStats, RawTermDictionary,
};
use postings::raw::{write_u64_u32_segment_from_index_seekable_to, RawSegmentFile};
use postings::PostingsIndex;

const SEAL_AFTER_DOCS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = [
        doc(
            10,
            "learned sparse retrieval",
            &["learned", "sparse", "retrieval", "retrieval"],
        ),
        doc(
            11,
            "bm25 segment pruning",
            &["bm25", "segment", "pruning", "search"],
        ),
        doc(
            12,
            "dictionary sidecars",
            &["dictionary", "sidecar", "raw", "segment"],
        ),
        doc(
            13,
            "live shard updates",
            &["live", "shard", "search", "updates"],
        ),
        doc(
            14,
            "raw file search",
            &["raw", "file", "search", "retrieval"],
        ),
        doc(
            15,
            "streaming ingestion",
            &["streaming", "ingestion", "raw", "bm25"],
        ),
        doc(
            16,
            "unsealed tail",
            &["live", "retrieval", "tail", "search"],
        ),
    ];

    let dir = tempfile::tempdir()?;
    let mut dictionary = RawTermDictionary::new();
    let mut live = PostingsIndex::new();
    let mut live_docs = 0usize;
    let mut sealed_paths = Vec::new();

    for document in &corpus {
        let terms = dictionary.encode_document(document.terms)?;
        live.add_weighted_document(document.id, &terms)?;
        live_docs += 1;

        if live_docs == SEAL_AFTER_DOCS {
            let path = dir
                .path()
                .join(format!("generation-{}.raw", sealed_paths.len()));
            let mut file = std::fs::File::create(&path)?;
            write_u64_u32_segment_from_index_seekable_to(&live, &mut file)?;
            file.sync_all()?;
            drop(file);

            sealed_paths.push(path);
            live = PostingsIndex::new();
            live_docs = 0;
        }
    }

    let dictionary_path = dir.path().join("dictionary.txt");
    write_dictionary_sidecar(&dictionary_path, &dictionary)?;

    let persisted_terms = std::fs::read_to_string(&dictionary_path)?;
    let loaded_dictionary = RawTermDictionary::from_terms_in_id_order(persisted_terms.lines())?;
    let query = loaded_dictionary.encode_query(["raw", "retrieval", "search"]);
    let mut sealed_segments = sealed_paths
        .iter()
        .map(RawSegmentFile::open)
        .collect::<Result<Vec<_>, _>>()?;
    let mut segment_refs: Vec<_> = sealed_segments.iter_mut().collect();
    let stats = RawBm25CorpusStats::from_raw_files_and_index(&mut segment_refs, &live, &query)?;
    let result = retrieve_bm25_raw_files_and_index_with_diagnostics(
        &mut segment_refs,
        &live,
        &query,
        5,
        Bm25Params::default(),
        &stats,
    )?;

    println!("dictionary terms: {}", loaded_dictionary.len());
    println!("sealed raw files: {}", sealed_paths.len());
    println!("live docs: {}", live_docs);
    println!(
        "sealed files seen/scored/pruned: {}/{}/{}",
        result.diagnostics.segments.segments_seen,
        result.diagnostics.segments.segments_scored,
        result.diagnostics.segments.segments_pruned
    );
    println!("top BM25 hits:");
    for (doc_id, score) in result.hits {
        println!("  doc {doc_id}: {score:.6}  {}", title(doc_id, &corpus));
    }

    Ok(())
}

fn write_dictionary_sidecar(
    path: &std::path::Path,
    dictionary: &RawTermDictionary,
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for (_, term) in dictionary.terms() {
        writeln!(writer, "{term}")?;
    }
    writer.flush()
}

struct Document {
    id: u32,
    title: &'static str,
    terms: &'static [&'static str],
}

fn doc(id: u32, title: &'static str, terms: &'static [&'static str]) -> Document {
    Document { id, title, terms }
}

fn title(doc_id: u32, corpus: &[Document]) -> &'static str {
    corpus
        .iter()
        .find(|doc| doc.id == doc_id)
        .map_or("unknown", |doc| doc.title)
}
