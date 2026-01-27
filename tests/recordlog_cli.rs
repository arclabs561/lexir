//! CLI contract tests for `lexir` recordlog commands.
#![cfg(feature = "cli")]

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use std::fs;

fn lexir() -> assert_cmd::Command {
    cargo_bin_cmd!("lexir")
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
