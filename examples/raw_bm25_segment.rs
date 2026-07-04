//! BM25 over a file-backed postings raw segment, with lexical terms encoded
//! through `RawTermDictionary` and sealed from a live numeric postings shard.
//!
//! Run:
//! `cargo run --example raw_bm25_segment --features raw-segment`

use lexir::bm25::Bm25Params;
use lexir::raw::{retrieve_bm25_raw_file, RawTermDictionary};
use postings::raw::{write_u64_u32_segment_from_index_seekable_to, RawSegmentFile};
use postings::PostingsIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = [
        doc(
            1,
            "learned sparse retrieval",
            &["neural", "neural", "neural", "retrieval"],
        ),
        doc(
            2,
            "lexical search",
            &["neural", "search", "search", "search", "search"],
        ),
        doc(3, "impact pruning", &["retrieval", "retrieval", "search"]),
    ];

    let mut dictionary = RawTermDictionary::new();
    let encoded_docs: Vec<_> = corpus
        .iter()
        .map(|doc| {
            dictionary
                .encode_document(doc.terms)
                .map(|terms| (doc.id, terms))
        })
        .collect::<Result<_, _>>()?;
    let mut live_shard = PostingsIndex::new();
    for (doc_id, terms) in &encoded_docs {
        live_shard.add_weighted_document(*doc_id, terms)?;
    }

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("lexir.raw");
    let mut file = std::fs::File::create(&path)?;
    write_u64_u32_segment_from_index_seekable_to(&live_shard, &mut file)?;
    file.sync_all()?;
    drop(file);

    let query = dictionary.encode_query(["neural", "search"]);
    let mut segment = RawSegmentFile::open(&path)?;
    let hits = retrieve_bm25_raw_file(&mut segment, &query, 3, Bm25Params::default())?;

    println!("dictionary:");
    for (term_id, term) in dictionary.terms() {
        println!("  {term_id}: {term}");
    }
    println!("query ids: {query:?}");
    println!("top BM25 hits:");
    for (doc_id, score) in hits {
        println!("  doc {doc_id}: {score:.6}  {}", title(doc_id, &corpus));
    }

    Ok(())
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
