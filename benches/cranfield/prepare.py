import re
import sys
from pathlib import Path

raw, data = map(Path, sys.argv[1:])
data.mkdir(exist_ok=True, parents=True)

def records(filename):
    text = (raw / filename).read_text(encoding="latin1")
    for m in re.finditer(r"^\.I (\d+)[ \t]*\n(.*?)(?=^\.I |\Z)", text, re.S | re.M):
        fields = re.split(r"^\.([A-Z])[ \t]*\n", m[2], flags=re.M)
        yield int(m[1]), {k: " ".join(v.split()) for k, v in zip(fields[1::2], fields[2::2])}

with (data/"corpus.tsv").open("w") as f:
    for doc, fields in records("cran.all.1400"):
        abstract = fields["W"]
        title = fields["T"]
        if not abstract.startswith(title):
            abstract = title + " " + abstract
        print(doc, abstract, sep="\t", file=f)
with (data/"queries.tsv").open("w") as f:
    for ordinal, (_, fields) in enumerate(records("cran.qry"), 1):
        print(ordinal, fields["W"], sep="\t", file=f)
