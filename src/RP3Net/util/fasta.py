import gzip
import re
from .util import resolve

RE_FASTA_HEADER = re.compile(r'^>([\w\-.:#*]+)') # https://www.ncbi.nlm.nih.gov/genbank/fastaformat/

def iter_fasta(filename):
    filename = resolve(filename)
    is_gzip = filename.suffix == '.gz'
    fasta_id, sequence = None, None
    with gzip.open(filename, 'rt') if is_gzip else open(filename) as f:
        for line in f:
            line = line.strip()
            m = RE_FASTA_HEADER.match(line)
            if m:
                if fasta_id is not None:
                    yield fasta_id, ''.join(sequence)
                sequence = []
                fasta_id = m.group(1)
            else:
                sequence.append(line)
        if fasta_id is not None:
            yield fasta_id, ''.join(sequence)

def read_fasta(filename):
    return {id: seq for id, seq in iter_fasta(filename)}