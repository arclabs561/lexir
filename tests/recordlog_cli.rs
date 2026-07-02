//! CLI contract tests for `lexir` recordlog commands.
#![cfg(feature = "cli")]

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use std::fs;

fn lexir() -> assert_cmd::Command {
    cargo_bin_cmd!("lexir")
}

#[test]
fn index_command_saves_searchable_streamed_corpus() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus.txt");
    let index = tmp.path().join("index.bin");
    fs::write(&corpus, "alpha one\nbeta two\nalpha two\n").expect("write corpus");

    lexir()
        .args([
            "index",
            "--input",
            corpus.to_str().unwrap(),
            "--output",
            index.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed 3 documents"));

    lexir()
        .args([
            "search-index",
            "--index",
            index.to_str().unwrap(),
            "--",
            "alpha",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Doc 0").and(predicate::str::contains("Doc 2")));
}

#[test]
fn search_command_indexes_corpus_path_directly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus.txt");
    fs::write(&corpus, "alpha one\nbeta two\nalpha two\n").expect("write corpus");

    lexir()
        .args(["search", "--input", corpus.to_str().unwrap(), "--", "beta"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Doc 1"));
}

#[test]
fn log_doctor_fix_repairs_missing_meta() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    lexir()
        .args([
            "log-add",
            "--root",
            root.to_str().unwrap(),
            "--doc-id",
            "1",
            "--text",
            "hello world",
        ])
        .assert()
        .success();

    lexir()
        .args([
            "log-add",
            "--root",
            root.to_str().unwrap(),
            "--doc-id",
            "2",
            "--text",
            "hello rust",
        ])
        .assert()
        .success();

    // Break meta.
    fs::remove_file(root.join("index.bin.meta")).expect("remove meta");

    // Without fix, doctor should fail.
    lexir()
        .args(["log-doctor", "--root", root.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("meta missing"));

    // With fix, doctor repairs.
    lexir()
        .args(["log-doctor", "--root", root.to_str().unwrap(), "--fix"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ok: wrote meta"));

    // And validate passes.
    lexir()
        .args(["log-validate", "--root", root.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn log_prune_preserves_validate() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    for (id, text) in [("1", "a b"), ("2", "b c"), ("3", "c d")] {
        lexir()
            .args([
                "log-add",
                "--root",
                root.to_str().unwrap(),
                "--doc-id",
                id,
                "--text",
                text,
            ])
            .assert()
            .success();
    }

    lexir()
        .args(["log-validate", "--root", root.to_str().unwrap()])
        .assert()
        .success();

    lexir()
        .args(["log-prune", "--root", root.to_str().unwrap()])
        .assert()
        .success();

    lexir()
        .args(["log-validate", "--root", root.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn log_compact_preserves_deletes_and_search() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    for (id, text) in [("1", "alpha live"), ("2", "alpha stale")] {
        lexir()
            .args([
                "log-add",
                "--root",
                root.to_str().unwrap(),
                "--doc-id",
                id,
                "--text",
                text,
            ])
            .assert()
            .success();
    }

    lexir()
        .args([
            "log-delete",
            "--root",
            root.to_str().unwrap(),
            "--doc-id",
            "2",
        ])
        .assert()
        .success();

    lexir()
        .args(["log-compact", "--root", root.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ok: compacted log (old_records=3 new_records=1)",
        ));

    lexir()
        .args(["log-validate", "--root", root.to_str().unwrap()])
        .assert()
        .success();

    lexir()
        .args([
            "log-search",
            "--root",
            root.to_str().unwrap(),
            "--",
            "alpha",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Doc 1").and(predicate::str::contains("Doc 2").not()));
}

#[test]
fn log_compact_removes_log_when_all_docs_are_deleted() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    lexir()
        .args([
            "log-add",
            "--root",
            root.to_str().unwrap(),
            "--doc-id",
            "1",
            "--text",
            "alpha stale",
        ])
        .assert()
        .success();

    lexir()
        .args([
            "log-delete",
            "--root",
            root.to_str().unwrap(),
            "--doc-id",
            "1",
        ])
        .assert()
        .success();

    lexir()
        .args(["log-compact", "--root", root.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ok: compacted log (old_records=2 new_records=0)",
        ));

    assert!(!root.join("ops.log").exists());

    lexir()
        .args(["log-validate", "--root", root.to_str().unwrap()])
        .assert()
        .success();

    lexir()
        .args(["log-status", "--root", root.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("log: missing (records=0)").and(
            predicate::str::contains("meta: present (applied_records=0, pending_records=0)"),
        ));
}

#[test]
fn torn_tail_best_effort_scan_succeeds_strict_fails() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    for (id, text) in [("1", "a b"), ("2", "b c"), ("3", "c d")] {
        lexir()
            .args([
                "log-add",
                "--root",
                root.to_str().unwrap(),
                "--doc-id",
                id,
                "--text",
                text,
            ])
            .assert()
            .success();
    }

    // Tear the tail (simulate crash during append).
    let log_path = root.join("ops.log");
    let bytes = fs::read(&log_path).expect("read ops.log");
    fs::write(&log_path, &bytes[..bytes.len().saturating_sub(3)]).expect("truncate ops.log");

    // Best-effort scan should succeed (returns prefix).
    lexir()
        .args(["log-scan", "--root", root.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("ok: scanned log"));

    // Strict scan must fail.
    lexir()
        .args(["log-scan", "--root", root.to_str().unwrap(), "--strict"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("UnexpectedEof"));
}
