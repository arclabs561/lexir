"""Closed-loop Cranfield replay with one BM25 index per client and a shared CPU encoder."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import random
from subprocess import PIPE, Popen
from threading import Barrier
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

def encode(texts, session, tokenizer):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    embeddings = np.empty((len(texts), 384), dtype=np.float32)
    for start in range(0, len(order), 4):
        batch = order[start:start + 4]
        ids = [e.ids for e in tokenizer.encode_batch([texts[i] for i in batch])]
        length = max(map(len, ids))
        inp = np.zeros((len(batch), length), dtype=np.int64)
        mask = np.zeros_like(inp)
        for j, row in enumerate(ids):
            inp[j, :len(row)] = row
            mask[j, :len(row)] = 1
        vec = session.run(None, dict(input_ids=inp, attention_mask=mask,
                                     token_type_ids=np.zeros_like(inp)))[0][:, 0, :]
        embeddings[batch] = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    assert np.isfinite(embeddings).all()
    return embeddings


def read_records(path):
    return dict(line.split("\t", 1) for line in path.read_text().splitlines())


def run(args):
    queries = read_records(args.data / "queries.tsv")
    corpus = read_records(args.data / "corpus.tsv")
    assert len(queries) == 225 and set(map(int, corpus)) == set(range(1, 1401))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    model = ort.InferenceSession(str(args.model / "model_quantized.onnx"), options,
                                 providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    vectors = encode([corpus[str(i)] for i in range(1, len(corpus) + 1)], model, tokenizer)
    order = list(queries) * 3
    random.Random(561).shuffle(order)
    expected = {}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    with (args.out / "latency.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(["clients", "method", "query", "milliseconds"])
        for clients in args.clients:
            for method in ("bm25", "bm25+bge"):
                gate = Barrier(clients)
                def worker(engine, subset):
                    def search(qid):
                        query = queries[qid]
                        start = perf_counter()
                        engine.stdin.write(query + "\n")
                        engine.stdin.flush()
                        pool = np.array([int(x) for x in engine.stdout.readline().split()])
                        if len(pool) != 100 or len(set(pool)) != 100 or min(pool) < 1 or max(pool) > len(corpus):
                            raise ValueError("Invalid top-100 BM25 candidates")
                        if method == "bm25+bge":
                            query_vector = encode(["Represent this sentence for searching relevant passages: " + query], model, tokenizer)[0]
                            scores = vectors[pool - 1] @ query_vector
                            pool = pool[np.lexsort((pool, -scores))]
                        return qid, tuple(pool[:10]), (perf_counter() - start) * 1000
                    for qid in order[:10]:
                        search(qid)
                    gate.wait(timeout=120)
                    start = perf_counter()
                    rows = [search(qid) for qid in subset]
                    return start, perf_counter(), rows
                with ExitStack() as stack:
                    engines = []
                    for _ in range(clients):
                        engine = stack.enter_context(Popen([args.binary, args.data / "corpus.tsv"],
                                                          stdin=PIPE, stdout=PIPE, text=True))
                        stack.callback(engine.stdin.close)
                        if engine.stdout.readline().strip() != "ready":
                            raise RuntimeError("BM25 index failed to start")
                        engines.append(engine)
                    with ThreadPoolExecutor(clients) as executor:
                        batches = list(executor.map(worker, engines, [order[i::clients] for i in range(clients)]))
                times = []
                for _, _, rows in batches:
                    for qid, hits, ms in rows:
                        if expected.setdefault((method, qid), hits) != hits:
                            raise ValueError(f"Ranking changed for query {qid}")
                        writer.writerow([clients, method, qid, ms])
                        times.append(ms)
                elapsed = max(b[1] for b in batches) - min(b[0] for b in batches)
                summary.append(dict(clients=clients, method=method, requests=len(times),
                                    qps=len(times)/elapsed, p50_ms=float(np.percentile(times, 50)),
                                    p95_ms=float(np.percentile(times, 95)), p99_ms=float(np.percentile(times, 99))))
                print(summary[-1], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "binary", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 2, 4])
    run(parser.parse_args())
