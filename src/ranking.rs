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

pub(crate) fn top_k_non_nan_scored_docs<I>(docs: I, k: usize) -> Vec<(u32, f32)>
where
    I: IntoIterator<Item = (u32, f32)>,
{
    top_k_scored_docs(docs, k, |score| !score.is_nan())
}

#[cfg(test)]
mod tests {
    use super::top_k_non_nan_scored_docs;

    #[test]
    fn non_nan_top_k_keeps_negative_infinity_after_finite_scores() {
        let ranked = top_k_non_nan_scored_docs(
            [(3, f32::NEG_INFINITY), (2, -1.0), (1, f32::NAN), (4, -1.0)],
            3,
        );

        assert_eq!(ranked, vec![(2, -1.0), (4, -1.0), (3, f32::NEG_INFINITY)]);
    }
}
