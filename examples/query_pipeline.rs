//! Full IR query pipeline: qexpr -> qplan -> lexir BM25 retrieval.
//!
//! Demonstrates:
//! 1. Building a lexir BM25 index from tokenized documents
//! 2. Constructing typed queries with `qexpr`
//! 3. Compiling queries into conjunctive plans with `qplan`
//! 4. Using the plan's `bag_terms` for BM25 retrieval
//! 5. Unsupported operators (Or, Not) produce explicit errors

use lexir::bm25::{Bm25Params, InvertedIndex};
use qexpr::{Phrase, QExpr, Term};
use qplan::compile_conjunctive;

fn main() {
    // -- Build the index ----------------------------------------------------------

    let mut index = InvertedIndex::new();

    let docs: &[(u32, &[&str])] = &[
        (0, &["rust", "memory", "safety", "ownership", "borrow"]),
        (1, &["rust", "async", "runtime", "tokio", "futures"]),
        (2, &["python", "machine", "learning", "numpy", "pandas"]),
        (3, &["rust", "wasm", "web", "assembly", "browser"]),
        (4, &["python", "web", "django", "rest", "api"]),
        (5, &["rust", "systems", "programming", "performance"]),
        (6, &["machine", "learning", "deep", "neural", "network"]),
        (7, &["web", "assembly", "wasm", "rust", "performance"]),
        (8, &["python", "async", "asyncio", "concurrency"]),
        (9, &["rust", "embedded", "systems", "no_std", "memory"]),
    ];

    for &(doc_id, terms) in docs {
        let owned: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
        index.add_document(doc_id, &owned);
    }

    println!("Indexed {} documents\n", index.num_docs());

    let params = Bm25Params::default();

    // -- Query 1: single term -----------------------------------------------------

    let q1 = QExpr::Term(Term::new("rust"));
    run_query("Term(rust)", &q1, &index, params);

    // -- Query 2: phrase ----------------------------------------------------------

    let q2 = QExpr::Phrase(Phrase::new(vec![
        Term::new("machine"),
        Term::new("learning"),
    ]));
    run_query("Phrase(machine learning)", &q2, &index, params);

    // -- Query 3: AND of term + phrase --------------------------------------------

    let q3 = QExpr::And(vec![
        QExpr::Term(Term::new("rust")),
        QExpr::Term(Term::new("performance")),
    ]);
    run_query("And(rust, performance)", &q3, &index, params);

    // -- Query 4: nested AND with phrase ------------------------------------------

    let q4 = QExpr::And(vec![
        QExpr::Phrase(Phrase::new(vec![Term::new("web"), Term::new("assembly")])),
        QExpr::Term(Term::new("rust")),
    ]);
    run_query("And(Phrase(web assembly), rust)", &q4, &index, params);

    // -- Unsupported operators produce errors -------------------------------------

    println!("--- Unsupported operators ---\n");

    let q_or = QExpr::Or(vec![
        QExpr::Term(Term::new("rust")),
        QExpr::Term(Term::new("python")),
    ]);
    match compile_conjunctive(&q_or) {
        Err(e) => println!("Or:    {e}"),
        Ok(_) => println!("Or:    unexpectedly compiled"),
    }

    let q_not = QExpr::Not(Box::new(QExpr::Term(Term::new("python"))));
    match compile_conjunctive(&q_not) {
        Err(e) => println!("Not:   {e}"),
        Ok(_) => println!("Not:   unexpectedly compiled"),
    }

    println!();
}

fn run_query(label: &str, expr: &QExpr, index: &InvertedIndex, params: Bm25Params) {
    println!("Query: {label}");

    let plan = match compile_conjunctive(expr) {
        Ok(p) => p,
        Err(e) => {
            println!("  compile error: {e}\n");
            return;
        }
    };

    println!("  Plan: bag_terms={:?}", plan.bag_terms);
    if !plan.phrases.is_empty() {
        println!("        phrases={:?}", plan.phrases);
    }
    println!("        bag_only={}", plan.is_bag_only());

    let results = index.retrieve(&plan.bag_terms, 5, params);
    match results {
        Ok(hits) => {
            println!("  Results ({} hits):", hits.len());
            for (doc_id, score) in &hits {
                println!("    doc {doc_id}: {score:.4}");
            }
        }
        Err(e) => println!("  retrieve error: {e}"),
    }
    println!();
}
