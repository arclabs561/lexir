#![cfg(feature = "raw-segment")]
//! Damaged supported raw segments must fail before invalid scoring assumptions;
//! valid legacy and current formats still reach the public BM25 consumer.
use lexir::raw::retrieve_bm25_raw_segment;
use postings::raw::{write_u64_u32_segment, RawDocument, RawSegment};
fn u64at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}
fn fixture() -> Vec<u8> {
    let terms = [(7, 1)];
    write_u64_u32_segment(&[RawDocument::new(0, &terms), RawDocument::new(2, &terms)]).unwrap()
}
fn v3(b: &[u8]) -> Vec<u8> {
    let footer = u64at(b, 64);
    // One term, one block: v4 integrity is four section/header CRCs plus block CRC.
    let integrity = footer - 20;
    let mut out = b[..integrity + 12].to_vec();
    out.extend_from_slice(&b[integrity + 16..]);
    out[8..12].copy_from_slice(&3u32.to_le_bytes());
    out[64..72].copy_from_slice(&((footer - 4) as u64).to_le_bytes());
    let n = out.len();
    out[n - 4..].copy_from_slice(&3u32.to_le_bytes());
    out
}
#[test]
fn valid_v3_and_v4_score_both_documents() {
    let clean = fixture();
    for bytes in [clean.clone(), v3(&clean)] {
        let seg = RawSegment::open(&bytes).unwrap();
        let hits = retrieve_bm25_raw_segment(&seg, &[7], 2, Default::default()).unwrap();
        assert_eq!(hits.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 2]);
    }
}
#[test]
fn v4_rejects_corrupt_header_maximum() {
    let mut b = fixture();
    b[24..28].copy_from_slice(&1u32.to_le_bytes());
    assert!(RawSegment::open(&b).is_err());
}
#[test]
fn v3_header_maximum_must_not_panic_public_scoring() {
    let mut b = v3(&fixture());
    b[24..28].copy_from_slice(&1u32.to_le_bytes());
    let seg = match RawSegment::open(&b) {
        Err(postings::raw::Error::InvalidLayout { reason }) => {
            eprintln!("malformed v3 rejected at RawSegment::open: {reason}");
            return;
        }
        Err(other) => panic!("unexpected open error: {other}"),
        Ok(seg) => seg,
    };
    assert_eq!(seg.meta().max_doc_id(), 1);
    let outcome =
        std::panic::catch_unwind(|| retrieve_bm25_raw_segment(&seg, &[7], 2, Default::default()));
    assert!(
        outcome.is_ok(),
        "public scoring panicked on accepted malformed segment"
    );
    assert!(
        outcome.unwrap().is_err(),
        "inconsistent ID metadata must return an error"
    );
}
