
import RP3Net as rp3
import tempfile
import urllib.request
import polars as pl
import itertools
import torch
import torchmetrics as tm
import unittest

class RP3PublicCheckpointTest(unittest.TestCase):
    def test_public_checkpoint(self):
        cp_url = 'https://ftp.ebi.ac.uk/pub/software/RP3Net/v0.1/checkpoints/rp3net_v0.1_d.ckpt'
        _, cp_file = tempfile.mkstemp(prefix='rp3_', suffix='.ckpt')
        print(f'Downloading checkpoint from {cp_url} to {cp_file}')
        urllib.request.urlretrieve(cp_url, cp_file)
        experimental_fasta_url = 'https://ftp.ebi.ac.uk/pub/software/RP3Net/publication/experimental_validation/constructs_protein.fasta.gz'
        _, experimental_fasta_file = tempfile.mkstemp(prefix='rp3_experimental_', suffix='.fasta.gz')
        print(f'Downloading experimental fasta from {experimental_fasta_url} to {experimental_fasta_file}')
        urllib.request.urlretrieve(experimental_fasta_url, experimental_fasta_file)
        experimental_pqt_url = 'https://ftp.ebi.ac.uk/pub/software/RP3Net/publication/experimental_validation/construct_expression.pqt'
        _, experimental_pqt_file = tempfile.mkstemp(prefix='rp3_experimental_', suffix='.pqt')
        print(f'Downloading experimental pqt from {experimental_pqt_url} to {experimental_pqt_file}')
        urllib.request.urlretrieve(experimental_pqt_url, experimental_pqt_file)
        
        m = rp3.load_model(rp3.RP3_DEFAULT_CONFIG, cp_file)
        df = pl.read_parquet(experimental_pqt_file).with_columns(yield_binary=pl.col('experiment_1').eq('Passed') | pl.col('experiment_2').eq('Passed'))
        seq_map = rp3.util.fasta.read_fasta(experimental_fasta_file)
        df = df.with_columns(seq=pl.col('id').replace_strict(seq_map), rp3_score=float('nan'))
        for ix in itertools.batched(range(df.shape[0]), 8):
            logits: torch.Tensor = m.predict(df[ix, 'seq'].to_list(), device='cpu')
            df[ix, 'rp3_score'] = logits.tolist()
        m_auroc = tm.classification.BinaryAUROC()
        auroc = m_auroc(torch.tensor(df.select('rp3_score').to_numpy()), torch.tensor(df.select('yield_binary').to_numpy()))
        self.assertGreater(auroc, 0.81)
