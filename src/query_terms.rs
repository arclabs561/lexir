pub(crate) fn term_multiplicities(query_terms: &[String]) -> Vec<(&str, usize)> {
    let mut counts: Vec<(&str, usize)> = Vec::with_capacity(query_terms.len());

    for term in query_terms {
        let term = term.as_str();
        if let Some((_, count)) = counts.iter_mut().find(|(seen, _)| *seen == term) {
            *count += 1;
        } else {
            counts.push((term, 1));
        }
    }

    counts
}
