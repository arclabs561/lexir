use lexir::bm25::{Bm25Params, InvertedIndex};
use std::error::Error;
use std::io::{self, BufRead, Write};

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let corpus = std::fs::read_to_string(std::env::args().nth(1).ok_or("missing corpus path")?)?;
    let mut index = InvertedIndex::new();
    for line in corpus.lines() {
        let (id, text) = line.split_once('\t').ok_or("invalid corpus line")?;
        index.add_document(id.parse()?, &tokens(text));
    }
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    writeln!(out, "ready")?;
    out.flush()?;
    for line in io::stdin().lock().lines() {
        let hits = index.retrieve(&tokens(&line?), 100, Bm25Params::default())?;
        for (id, _) in hits {
            write!(out, "{} ", id)?;
        }
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}
