//! BM25 over a file-backed postings raw segment.
//!
//! Run:
//! `cargo run --example raw_bm25_segment --features raw-segment`

use lexir::bm25::Bm25Params;
use lexir::raw::retrieve_bm25_raw_file;
use postings::raw::{write_u64_u32_segment_sorted_from_iter_to, RawDocument, RawSegmentFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = [
        RawDocument::new(1, &[(10, 3), (20, 1)]),
        RawDocument::new(2, &[(10, 1), (30, 4)]),
        RawDocument::new(3, &[(20, 2), (30, 1)]),
    ];
    let path = std::env::temp_dir().join(format!("lexir-raw-{}.segment", std::process::id()));
    let mut file = std::fs::File::create(&path)?;
    write_u64_u32_segment_sorted_from_iter_to(docs, &mut file)?;
    drop(file);

    let mut segment = RawSegmentFile::open(&path)?;
    let hits = retrieve_bm25_raw_file(&mut segment, &[10, 30], 3, Bm25Params::default())?;

    for (doc_id, score) in hits {
        println!("{doc_id}\t{score:.6}");
    }

    std::fs::remove_file(path)?;
    Ok(())
}
