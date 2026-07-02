#[inline]
fn cmp_doc_scores(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

fn top_k_scored_docs<I, F>(docs: I, k: usize, keep: F) -> Vec<(u32, f32)>
where
    I: IntoIterator<Item = (u32, f32)>,
    F: Fn(f32) -> bool,
{
    if k == 0 {
        return Vec::new();
    }

    let mut ranked: Vec<(u32, f32)> = Vec::with_capacity(k);
    let mut sorted = false;
    for doc in docs {
        if !keep(doc.1) {
            continue;
        }
        if ranked.len() < k {
            ranked.push(doc);
            continue;
        }
        if !sorted {
            ranked.sort_by(cmp_doc_scores);
            sorted = true;
        }
        if cmp_doc_scores(&doc, ranked.last().expect("top-k buffer is full")).is_lt() {
            let last = ranked.len() - 1;
            ranked[last] = doc;
            let mut i = last;
            while i > 0 && cmp_doc_scores(&ranked[i], &ranked[i - 1]).is_lt() {
                ranked.swap(i, i - 1);
                i -= 1;
            }
        }
    }
    if !sorted {
        ranked.sort_by(cmp_doc_scores);
    }
    ranked
}

pub(crate) fn top_k_positive_scored_docs<I>(docs: I, k: usize) -> Vec<(u32, f32)>
where
    I: IntoIterator<Item = (u32, f32)>,
{
    top_k_scored_docs(docs, k, |score| score.is_finite() && score > 0.0)
}

pub(crate) fn top_k_finite_scored_docs<I>(docs: I, k: usize) -> Vec<(u32, f32)>
where
    I: IntoIterator<Item = (u32, f32)>,
{
    top_k_scored_docs(docs, k, f32::is_finite)
}
