//! `lexir` CLI: BM25 playground (in-memory + persisted index).

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};
#[cfg(feature = "cli")]
use durability::recordlog::{RecordLogReadMode, RecordLogReader, RecordLogWriter};
#[cfg(feature = "cli")]
use durability::{Directory, DurableDirectory};
#[cfg(feature = "cli")]
use lexir::bm25::{Bm25Params, InvertedIndex};
#[cfg(feature = "cli")]
use std::collections::BTreeMap;
#[cfg(feature = "cli")]
use std::io::Read;
#[cfg(feature = "cli")]
use std::path::Path;
#[cfg(feature = "cli")]
use std::path::PathBuf;
#[cfg(feature = "cli")]
use std::sync::Arc;

#[cfg(feature = "cli")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum LogOp {
    Add { doc_id: u32, terms: Vec<String> },
    Delete { doc_id: u32 },
}

#[cfg(feature = "cli")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct LogMeta {
    /// Number of records in `ops.log` already included in the checkpoint.
    applied_records: usize,
}

#[cfg(feature = "cli")]
fn meta_path(checkpoint: &str) -> String {
    format!("{checkpoint}.meta")
}

#[cfg(feature = "cli")]
fn load_meta<D: Directory + ?Sized>(
    dir: &D,
    checkpoint: &str,
) -> Result<Option<LogMeta>, Box<dyn std::error::Error>> {
    let path = meta_path(checkpoint);
    if !dir.exists(&path) {
        return Ok(None);
    }
    let mut f = dir.open_file(&path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(Some(postcard::from_bytes::<LogMeta>(&bytes)?))
}

// Meta writes are done inline at call sites, so they share the same durability strength
// as the checkpoint itself (atomic vs durable barriers).

#[cfg(feature = "cli")]
fn apply_ops(idx: &mut InvertedIndex, ops: &[LogOp]) {
    for op in ops {
        match op {
            LogOp::Add { doc_id, terms } => idx.add_document(*doc_id, terms),
            LogOp::Delete { doc_id } => {
                let _ = idx.delete_document(*doc_id);
            }
        }
    }
}

#[cfg(feature = "cli")]
fn fingerprint(idx: &InvertedIndex) -> u64 {
    // Deterministic FNV-1a 64-bit hash over a stable traversal of the index.
    fn fnv1a_step(mut h: u64, bytes: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        if h == 0 {
            h = FNV_OFFSET;
        }
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    let mut h: u64 = 0;
    h = fnv1a_step(h, &idx.num_docs().to_le_bytes());

    let mut terms: Vec<&str> = idx.terms().collect();
    terms.sort_unstable();
    for t in terms {
        h = fnv1a_step(h, t.as_bytes());
        h = fnv1a_step(h, &idx.doc_frequency(t).to_le_bytes());
        let mut postings: Vec<(u32, u32)> = idx.postings_iter(t).collect();
        postings.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (doc_id, tf) in postings {
            h = fnv1a_step(h, &doc_id.to_le_bytes());
            h = fnv1a_step(h, &tf.to_le_bytes());
        }
    }
    h
}

#[cfg(feature = "cli")]
fn write_meta(
    dir: &durability::FsDirectory,
    checkpoint: &str,
    meta: &LogMeta,
    durable: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = postcard::to_allocvec(meta)?;
    let path = meta_path(checkpoint);
    if durable {
        dir.atomic_write_durable(&path, &bytes)?;
    } else {
        dir.atomic_write(&path, &bytes)?;
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn require_meta_or_abort(
    dir: &durability::FsDirectory,
    checkpoint: &str,
    log: &str,
    ops_len: usize,
) -> Result<LogMeta, Box<dyn std::error::Error>> {
    if let Some(meta) = load_meta(dir, checkpoint)? {
        if meta.applied_records > ops_len {
            return Err(format!(
                "invalid meta: applied_records={} > ops_len={}",
                meta.applied_records, ops_len
            )
            .into());
        }
        return Ok(meta);
    }
    if ops_len == 0 {
        return Ok(LogMeta::default());
    }
    if dir.exists(checkpoint) {
        return Err(format!(
            "missing checkpoint meta ({}) but found non-empty log ({}) with {} records.\n\
             Refusing to guess applied_records (would risk double-applying).\n\
             Fix: run `lexir log-checkpoint --checkpoint {checkpoint} --log {log} --reset-log` once.",
            meta_path(checkpoint),
            log,
            ops_len
        )
        .into());
    }
    Ok(LogMeta::default())
}

#[cfg(feature = "cli")]
fn read_ops_best_effort(
    dir: Arc<dyn Directory>,
    log: &str,
) -> Result<Vec<LogOp>, Box<dyn std::error::Error>> {
    if !dir.exists(log) {
        return Ok(Vec::new());
    }
    let reader = RecordLogReader::new(dir, log.to_string());
    Ok(reader.read_all_postcard(RecordLogReadMode::BestEffort)?)
}

#[cfg(feature = "cli")]
fn read_ops(
    dir: Arc<dyn Directory>,
    log: &str,
    mode: RecordLogReadMode,
) -> Result<Vec<LogOp>, Box<dyn std::error::Error>> {
    if !dir.exists(log) {
        return Ok(Vec::new());
    }
    let reader = RecordLogReader::new(dir, log.to_string());
    Ok(reader.read_all_postcard(mode)?)
}

#[cfg(feature = "cli")]
fn infer_applied_records(
    dir: &durability::FsDirectory,
    checkpoint: &str,
    _log: &str,
    ops: &[LogOp],
    max_scan: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    if !dir.exists(checkpoint) {
        return Ok(0);
    }
    if ops.is_empty() {
        return Ok(0);
    }
    if ops.len() > max_scan {
        return Err(format!(
            "refusing to infer applied_records: ops_len={} > max_scan={}",
            ops.len(),
            max_scan
        )
        .into());
    }

    let ckpt_idx = InvertedIndex::load(dir, checkpoint)?;
    let ckpt_fp = fingerprint(&ckpt_idx);

    // Compute final state fingerprint once.
    let mut idx_full = InvertedIndex::new();
    apply_ops(&mut idx_full, ops);
    let fp_full = fingerprint(&idx_full);

    if ckpt_fp == fp_full {
        // Checkpoint already reflects the full log replay; it is safe to treat the entire log as applied.
        return Ok(ops.len());
    }

    // Otherwise, find k such that replaying first k ops yields the checkpoint state.
    let mut idx_prefix = InvertedIndex::new();
    let mut last_match: Option<usize> = None;
    for (i, op) in ops.iter().enumerate() {
        apply_ops(&mut idx_prefix, std::slice::from_ref(op));
        if fingerprint(&idx_prefix) == ckpt_fp {
            last_match = Some(i + 1);
        }
    }

    let Some(k) = last_match else {
        return Err(format!(
            "could not infer applied_records: checkpoint does not match any prefix of log replay (ops_len={})",
            ops.len()
        )
        .into());
    };

    // Validate that checkpoint + suffix(k..) yields full state.
    let mut idx_ckpt_suffix = InvertedIndex::load(dir, checkpoint)?;
    apply_ops(&mut idx_ckpt_suffix, &ops[k..]);
    let fp_ckpt_suffix = fingerprint(&idx_ckpt_suffix);
    if fp_ckpt_suffix != fp_full {
        return Err(format!(
            "inferred applied_records={} failed validation: checkpoint+suffix != full replay.\n\
             fp_checkpoint_suffix={fp_ckpt_suffix:#x}\n\
             fp_full_replay={fp_full:#x}",
            k
        )
        .into());
    }

    Ok(k)
}

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Lexical search CLI", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "cli")]
#[derive(Subcommand, Debug)]
enum Commands {
    /// Build an index from a corpus file (one document per line) and save it.
    Index {
        /// Path to a corpus file (one document per line).
        #[arg(short, long)]
        input: PathBuf,

        /// Output index file (relative to its parent dir).
        #[arg(short, long)]
        output: PathBuf,

        /// Use stable-storage durability barriers (fsync + parent-dir sync).
        ///
        /// This is only supported for filesystem-backed directories.
        #[arg(long, default_value_t = false)]
        durable: bool,
    },

    /// Search a saved index file.
    SearchIndex {
        /// Path to index file (produced by `index`).
        #[arg(short, long)]
        index: PathBuf,

        /// Top-k results to return.
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Query terms.
        query: Vec<String>,
    },

    /// Search a corpus file (one document per line).
    Search {
        /// Path to a corpus file (one document per line).
        #[arg(short, long)]
        input: PathBuf,

        /// Top-k results to return
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Query terms
        query: Vec<String>,
    },

    /// Append one document to a persistent log and update the checkpoint.
    ///
    /// This provides an incremental, replayable update stream for E2E validation:
    /// - checkpoint holds a snapshot (`index.bin`)
    /// - log holds append-only ops (`ops.log`)
    LogAdd {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Document id.
        #[arg(long)]
        doc_id: u32,

        /// Document text (tokenized by whitespace).
        #[arg(long)]
        text: String,

        /// Write checkpoint with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,
    },

    /// Delete one document via the record log and update the checkpoint.
    LogDelete {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Document id.
        #[arg(long)]
        doc_id: u32,

        /// Write checkpoint with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,
    },

    /// Load checkpoint + replay log, then search.
    LogSearch {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Top-k results to return.
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Query terms.
        query: Vec<String>,
    },

    /// Write a fresh checkpoint from checkpoint+log, then optionally clear the log.
    LogCheckpoint {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Clear the log after checkpointing.
        #[arg(long, default_value_t = false)]
        reset_log: bool,

        /// Write checkpoint with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Print a short status summary for checkpoint/log/meta.
    LogStatus {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Diagnose checkpoint/log/meta health and optionally fix missing meta.
    LogDoctor {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Maximum number of log records to scan when inferring `applied_records`.
        #[arg(long, default_value_t = 10_000)]
        max_scan: usize,

        /// If possible, write a missing meta file (safe inference + validation required).
        #[arg(long, default_value_t = false)]
        fix: bool,

        /// Write meta with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Compact the log by squashing to the current final state.
    ///
    /// This rewrites `ops.log` to contain at most one `Add` per surviving document (sorted by
    /// doc id), deletes redundant history, and writes a fresh checkpoint consistent with the
    /// compacted log. After this, `log-validate` should still pass.
    LogCompact {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Write checkpoint/log with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Prune log history while preserving replay equivalence.
    ///
    /// Note: `lexir`’s `log-validate` assumes the log is sufficient to rebuild state from empty.
    /// That means we cannot “drop the applied prefix” without also making the remaining log
    /// self-contained. So this command rewrites `ops.log` to a minimal equivalent log (one `Add`
    /// per surviving document), writes a fresh checkpoint, and updates meta accordingly.
    LogPrune {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Write log/meta with stable-storage durability barriers.
        #[arg(long, default_value_t = false)]
        durable: bool,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Scan the record log and report how many records are readable.
    ///
    /// This ignores checkpoint/meta and is useful for diagnosing torn tails:
    /// - best-effort should stop at a torn tail
    /// - strict should error
    LogScan {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Validate that checkpoint + suffix-replay matches full log replay.
    LogValidate {
        /// Root directory for storage.
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Checkpoint snapshot file (under root).
        #[arg(long, default_value = "index.bin")]
        checkpoint: String,

        /// Record log file (under root).
        #[arg(long, default_value = "ops.log")]
        log: String,

        /// Use strict log replay (treat any torn tail/corruption as an error).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cli")]
    {
        let args = Args::parse();

        match args.command {
            Commands::Index {
                input,
                output,
                durable,
            } => {
                let text = std::fs::read_to_string(&input)?;
                let mut idx = InvertedIndex::new();
                for (i, line) in text.lines().enumerate() {
                    let terms: Vec<String> =
                        line.split_whitespace().map(|s| s.to_string()).collect();
                    idx.add_document(i as u32, &terms);
                }

                let parent = output.parent().unwrap_or_else(|| Path::new("."));
                let file_name = output
                    .file_name()
                    .ok_or("invalid output path")?
                    .to_str()
                    .ok_or("invalid output name")?;

                let dir = durability::FsDirectory::new(parent)?;
                if durable {
                    idx.save_durable(&dir, file_name)?;
                } else {
                    idx.save(&dir, file_name)?;
                }

                println!("Indexed {} documents to {:?}", text.lines().count(), output);
            }
            Commands::SearchIndex { index, k, query } => {
                let parent = index.parent().unwrap_or_else(|| Path::new("."));
                let file_name = index
                    .file_name()
                    .ok_or("invalid index path")?
                    .to_str()
                    .ok_or("invalid index name")?;

                let dir = durability::FsDirectory::new(parent)?;
                let idx = InvertedIndex::load(&dir, file_name)?;

                let results = idx.retrieve(&query, k, Bm25Params::default())?;

                println!("Results for {:?}:", query);
                for (doc_id, score) in results {
                    println!("  Doc {}: score {:.4}", doc_id, score);
                }
            }
            Commands::Search { input, k, query } => {
                let text = std::fs::read_to_string(&input)?;
                let mut idx = InvertedIndex::new();
                for (i, line) in text.lines().enumerate() {
                    let terms: Vec<String> =
                        line.split_whitespace().map(|s| s.to_string()).collect();
                    idx.add_document(i as u32, &terms);
                }

                let results = idx.retrieve(&query, k, Bm25Params::default())?;

                println!("Results for {:?}:", query);
                for (doc_id, score) in results {
                    println!("  Doc {}: score {:.4}", doc_id, score);
                }
            }
            Commands::LogAdd {
                root,
                checkpoint,
                log,
                doc_id,
                text,
                durable,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let mut idx = if dir.exists(&checkpoint) {
                    InvertedIndex::load(&dir, &checkpoint)?
                } else {
                    InvertedIndex::new()
                };

                // Replay *suffix* ops (those not yet included in the checkpoint).
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let ops: Vec<LogOp> = read_ops_best_effort(arc.clone(), &log)?;
                let meta = require_meta_or_abort(&dir, &checkpoint, &log, ops.len())?;
                apply_ops(&mut idx, &ops[meta.applied_records..]);

                // Apply new op.
                let terms: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
                idx.add_document(doc_id, &terms);

                // Append to log.
                let mut w = RecordLogWriter::new(arc, log);
                w.append_postcard(&LogOp::Add { doc_id, terms })?;
                if durable {
                    w.flush_and_sync()?;
                } else {
                    w.flush()?;
                }

                // Write checkpoint.
                if durable {
                    idx.save_durable(&dir, &checkpoint)?;
                } else {
                    idx.save(&dir, &checkpoint)?;
                }

                // Mark all currently-present log records as included in the checkpoint.
                let meta = LogMeta {
                    applied_records: ops.len() + 1,
                };
                write_meta(&dir, &checkpoint, &meta, durable)?;

                println!("ok: wrote op + checkpoint (doc_id={doc_id})");
            }
            Commands::LogDelete {
                root,
                checkpoint,
                log,
                doc_id,
                durable,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let mut idx = if dir.exists(&checkpoint) {
                    InvertedIndex::load(&dir, &checkpoint)?
                } else {
                    InvertedIndex::new()
                };

                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let ops: Vec<LogOp> = read_ops_best_effort(arc.clone(), &log)?;
                let meta = require_meta_or_abort(&dir, &checkpoint, &log, ops.len())?;
                apply_ops(&mut idx, &ops[meta.applied_records..]);

                let existed = idx.delete_document(doc_id);

                let mut w = RecordLogWriter::new(arc, log);
                w.append_postcard(&LogOp::Delete { doc_id })?;
                if durable {
                    w.flush_and_sync()?;
                } else {
                    w.flush()?;
                }

                if durable {
                    idx.save_durable(&dir, &checkpoint)?;
                } else {
                    idx.save(&dir, &checkpoint)?;
                }

                let meta = LogMeta {
                    applied_records: ops.len() + 1,
                };
                write_meta(&dir, &checkpoint, &meta, durable)?;

                println!("ok: wrote op + checkpoint (doc_id={doc_id}, existed={existed})");
            }
            Commands::LogSearch {
                root,
                checkpoint,
                log,
                k,
                strict,
                query,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let mut idx = if dir.exists(&checkpoint) {
                    InvertedIndex::load(&dir, &checkpoint)?
                } else {
                    InvertedIndex::new()
                };

                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;
                let meta = require_meta_or_abort(&dir, &checkpoint, &log, ops.len())?;
                apply_ops(&mut idx, &ops[meta.applied_records..]);

                let results = idx.retrieve(&query, k, Bm25Params::default())?;
                println!("Results for {:?}:", query);
                for (doc_id, score) in results {
                    println!("  Doc {}: score {:.4}", doc_id, score);
                }
            }
            Commands::LogCheckpoint {
                root,
                checkpoint,
                log,
                reset_log,
                durable,
                strict,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let mut idx = if dir.exists(&checkpoint) {
                    InvertedIndex::load(&dir, &checkpoint)?
                } else {
                    InvertedIndex::new()
                };

                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;
                let meta = require_meta_or_abort(&dir, &checkpoint, &log, ops.len())?;
                apply_ops(&mut idx, &ops[meta.applied_records..]);

                if durable {
                    idx.save_durable(&dir, &checkpoint)?;
                } else {
                    idx.save(&dir, &checkpoint)?;
                }

                if reset_log {
                    // Reset log by deleting it; next writer will recreate with header.
                    if dir.exists(&log) {
                        dir.delete(&log)?;
                    }
                }

                let meta = if reset_log {
                    LogMeta { applied_records: 0 }
                } else {
                    LogMeta {
                        applied_records: ops.len(),
                    }
                };

                write_meta(&dir, &checkpoint, &meta, durable)?;

                println!("ok: checkpointed {} ops (reset_log={reset_log})", ops.len());
            }
            Commands::LogStatus {
                root,
                checkpoint,
                log,
                strict,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;
                let ops_len = ops.len();

                let ckpt_exists = dir.exists(&checkpoint);
                let log_exists = dir.exists(&log);
                let meta = load_meta(&dir, &checkpoint)?;

                let applied = match meta {
                    Some(m) => {
                        if m.applied_records > ops_len {
                            return Err(format!(
                                "invalid meta: applied_records={} > ops_len={}",
                                m.applied_records, ops_len
                            )
                            .into());
                        }
                        Some(m.applied_records)
                    }
                    None => None,
                };

                if ckpt_exists && ops_len > 0 && applied.is_none() {
                    return Err(format!(
                        "ambiguous: checkpoint exists and log is non-empty, but meta is missing ({})",
                        meta_path(&checkpoint)
                    )
                    .into());
                }

                let applied = applied.unwrap_or(0);
                let pending = ops_len.saturating_sub(applied);

                println!(
                    "checkpoint: {}",
                    if ckpt_exists { "present" } else { "missing" }
                );
                println!(
                    "log: {} (records={})",
                    if log_exists { "present" } else { "missing" },
                    ops_len
                );
                println!(
                    "meta: {} (applied_records={}, pending_records={})",
                    if dir.exists(&meta_path(&checkpoint)) {
                        "present"
                    } else {
                        "missing"
                    },
                    applied,
                    pending
                );
            }
            Commands::LogDoctor {
                root,
                checkpoint,
                log,
                max_scan,
                fix,
                durable,
                strict,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;

                let ckpt_exists = dir.exists(&checkpoint);
                let log_exists = dir.exists(&log);
                let meta_path = meta_path(&checkpoint);
                let meta = load_meta(&dir, &checkpoint)?;

                println!(
                    "checkpoint: {}",
                    if ckpt_exists { "present" } else { "missing" }
                );
                println!(
                    "log: {} (records={})",
                    if log_exists { "present" } else { "missing" },
                    ops.len()
                );
                println!(
                    "meta: {}",
                    if dir.exists(&meta_path) {
                        "present"
                    } else {
                        "missing"
                    }
                );

                match meta {
                    Some(m) => {
                        if m.applied_records > ops.len() {
                            return Err(format!(
                                "invalid meta: applied_records={} > ops_len={}",
                                m.applied_records,
                                ops.len()
                            )
                            .into());
                        }
                        println!(
                            "ok: meta is consistent (applied_records={}, pending_records={})",
                            m.applied_records,
                            ops.len().saturating_sub(m.applied_records)
                        );
                    }
                    None => {
                        if !ckpt_exists || ops.is_empty() {
                            println!("ok: meta missing but state is non-ambiguous");
                        } else {
                            println!(
                                "issue: meta missing and state is ambiguous (checkpoint exists + non-empty log)"
                            );

                            let inferred =
                                infer_applied_records(&dir, &checkpoint, &log, &ops, max_scan)?;
                            println!("inferred: applied_records={}", inferred);
                            println!(
                                "recommended: write meta via `lexir log-doctor --root <root> --checkpoint {checkpoint} --log {log} --fix`"
                            );

                            if fix {
                                let meta = LogMeta {
                                    applied_records: inferred,
                                };
                                write_meta(&dir, &checkpoint, &meta, durable)?;
                                println!("ok: wrote meta ({})", meta_path);
                            } else {
                                return Err(format!(
                                    "meta missing; rerun with `--fix` to write {}",
                                    meta_path
                                )
                                .into());
                            }
                        }
                    }
                }
            }
            Commands::LogCompact {
                root,
                checkpoint,
                log,
                durable,
                strict,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc.clone(), &log, mode)?;
                let old_ops_len = ops.len();

                // Track final state per doc id (None = deleted).
                let mut final_state: BTreeMap<u32, Option<Vec<String>>> = BTreeMap::new();
                for op in &ops {
                    match op {
                        LogOp::Add { doc_id, terms } => {
                            final_state.insert(*doc_id, Some(terms.clone()));
                        }
                        LogOp::Delete { doc_id } => {
                            final_state.insert(*doc_id, None);
                        }
                    }
                }

                let mut new_ops: Vec<LogOp> = Vec::new();
                for (doc_id, maybe_terms) in final_state {
                    if let Some(terms) = maybe_terms {
                        new_ops.push(LogOp::Add { doc_id, terms });
                    }
                }

                // Rewrite log (or delete it if empty).
                if new_ops.is_empty() {
                    if dir.exists(&log) {
                        dir.delete(&log)?;
                    }
                } else {
                    let tmp_log = format!("{log}.tmp");
                    if dir.exists(&tmp_log) {
                        dir.delete(&tmp_log)?;
                    }
                    let mut w = RecordLogWriter::new(arc.clone(), tmp_log.clone());
                    for op in &new_ops {
                        w.append_postcard(op)?;
                    }
                    if durable {
                        w.flush_and_sync()?;
                        dir.atomic_rename_durable(&tmp_log, &log)?;
                    } else {
                        w.flush()?;
                        dir.atomic_rename(&tmp_log, &log)?;
                    }
                }

                // Write a fresh checkpoint consistent with the compacted log.
                let mut idx = InvertedIndex::new();
                apply_ops(&mut idx, &new_ops);
                if durable {
                    idx.save_durable(&dir, &checkpoint)?;
                } else {
                    idx.save(&dir, &checkpoint)?;
                }

                let meta = LogMeta {
                    applied_records: new_ops.len(),
                };
                write_meta(&dir, &checkpoint, &meta, durable)?;

                println!(
                    "ok: compacted log (old_records={} new_records={})",
                    old_ops_len,
                    new_ops.len()
                );
            }
            Commands::LogPrune {
                root,
                checkpoint,
                log,
                durable,
                strict,
            } => {
                // Implemented as a “rebase/compact” to preserve `log-validate` semantics.
                let dir = durability::FsDirectory::new(&root)?;
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc.clone(), &log, mode)?;
                let old_ops_len = ops.len();

                // Track final state per doc id (None = deleted).
                let mut final_state: BTreeMap<u32, Option<Vec<String>>> = BTreeMap::new();
                for op in &ops {
                    match op {
                        LogOp::Add { doc_id, terms } => {
                            final_state.insert(*doc_id, Some(terms.clone()));
                        }
                        LogOp::Delete { doc_id } => {
                            final_state.insert(*doc_id, None);
                        }
                    }
                }

                let mut new_ops: Vec<LogOp> = Vec::new();
                for (doc_id, maybe_terms) in final_state {
                    if let Some(terms) = maybe_terms {
                        new_ops.push(LogOp::Add { doc_id, terms });
                    }
                }

                // Rewrite log (or delete it if empty).
                if new_ops.is_empty() {
                    if dir.exists(&log) {
                        dir.delete(&log)?;
                    }
                } else {
                    let tmp_log = format!("{log}.tmp");
                    if dir.exists(&tmp_log) {
                        dir.delete(&tmp_log)?;
                    }
                    let mut w = RecordLogWriter::new(arc.clone(), tmp_log.clone());
                    for op in &new_ops {
                        w.append_postcard(op)?;
                    }
                    if durable {
                        w.flush_and_sync()?;
                        dir.atomic_rename_durable(&tmp_log, &log)?;
                    } else {
                        w.flush()?;
                        dir.atomic_rename(&tmp_log, &log)?;
                    }
                }

                // Write a fresh checkpoint consistent with the rewritten log.
                let mut idx = InvertedIndex::new();
                apply_ops(&mut idx, &new_ops);
                if durable {
                    idx.save_durable(&dir, &checkpoint)?;
                } else {
                    idx.save(&dir, &checkpoint)?;
                }

                let meta = LogMeta {
                    applied_records: new_ops.len(),
                };
                write_meta(&dir, &checkpoint, &meta, durable)?;

                println!(
                    "ok: pruned history (old_records={} new_records={})",
                    old_ops_len,
                    new_ops.len()
                );
            }
            Commands::LogScan { root, log, strict } => {
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;
                println!("ok: scanned log (records={})", ops.len());
            }
            Commands::LogValidate {
                root,
                checkpoint,
                log,
                strict,
            } => {
                let dir = durability::FsDirectory::new(&root)?;
                let arc: Arc<dyn Directory> = Arc::new(durability::FsDirectory::new(&root)?);
                let mode = if strict {
                    RecordLogReadMode::Strict
                } else {
                    RecordLogReadMode::BestEffort
                };
                let ops: Vec<LogOp> = read_ops(arc, &log, mode)?;

                let meta = require_meta_or_abort(&dir, &checkpoint, &log, ops.len())?;

                let mut idx_ckpt = if dir.exists(&checkpoint) {
                    InvertedIndex::load(&dir, &checkpoint)?
                } else {
                    InvertedIndex::new()
                };
                apply_ops(&mut idx_ckpt, &ops[meta.applied_records..]);
                let fp_ckpt = fingerprint(&idx_ckpt);

                let mut idx_full = InvertedIndex::new();
                apply_ops(&mut idx_full, &ops);
                let fp_full = fingerprint(&idx_full);

                if fp_ckpt != fp_full {
                    return Err(format!(
                        "validate failed: checkpoint+suffix != full log replay.\n\
                         fp_checkpoint_suffix={fp_ckpt:#x}\n\
                         fp_full_replay={fp_full:#x}\n\
                         ops_len={} applied_records={}",
                        ops.len(),
                        meta.applied_records
                    )
                    .into());
                }

                println!(
                    "ok: validate passed (ops_len={}, applied_records={}, fp={:#x})",
                    ops.len(),
                    meta.applied_records,
                    fp_ckpt
                );
            }
        }
    }

    #[cfg(not(feature = "cli"))]
    println!("CLI feature is disabled. Build with --features cli to enable.");

    Ok(())
}
