import unittest
import tempfile
import RP3Net as rp3
import torch
import itertools
import torchmetrics as tm
import ml_collections as mlc
import polars as pl
import shutil


class RP3NetEbiTest(unittest.TestCase):
    def test_training_a(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_a.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])
        shutil.rmtree(rootdir, ignore_errors=True)

    def test_training_b(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_b.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])
        shutil.rmtree(rootdir, ignore_errors=True)

    def test_training_d(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_d.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])
        shutil.rmtree(rootdir, ignore_errors=True)

    def test_ebi_az_1(self):
        m = rp3.load_model(rp3.RP3_CONFIG_B, '/homes/evgeny/rp3/prod/v0.1/checkpoints/rp3net_v0.1_b_az_1.ckpt')
        score = m.predict(['PRTEINWQENCE', 'PRTEIN', 'SQWENCE'])
        self.assertEqual(score.shape, (3,))
        self.assertTrue((score > 0).all())
        self.assertTrue((score < 1).all())

    def test_model_d_100_constructs(self):
        m = rp3.load_model(rp3.RP3_DEFAULT_CONFIG, '/homes/evgeny/rp3/prod/v0.1/checkpoints/rp3net_v0.1_d.ckpt')
        m = m.to(device='cpu')
        m = m.eval()
        df = rp3.training.data.read_full_df_pl('/homes/evgeny/rp3/prod/publication/experimental_validation/construct_expression.csv.gz')
        seq_map = rp3.util.fasta.read_fasta('/homes/evgeny/rp3/prod/publication/experimental_validation/constructs_protein.fasta.gz')
        df = df.with_columns(seq=pl.col('id').replace_strict(seq_map), rp3_score=float('nan'))
        for ix in itertools.batched(range(df.shape[0]), 8):
            logits: torch.Tensor = m.predict(df[ix, 'seq'].to_list(), device='cpu')
            df[ix, 'rp3_score'] = logits.tolist()
        m_auroc = tm.classification.BinaryAUROC()
        auroc = m_auroc(torch.tensor(df.select('rp3_score').to_numpy()), torch.tensor(df.select('yield_binary').to_numpy()))
        self.assertGreater(auroc, 0.81)

    def test_fm_checkpoint(self):
        config = mlc.FrozenConfigDict({
            'fm':{
                'type': 'esm2_650m',
                'cp': '~/nobackup/hf-git/esm2_t33_650M_UR50D/pytorch_model.bin',
            },
            'aggregation': 'mean',
            'classification_head': {
                'embedding_dim': 1280,
                'bias': False,
                'end_bias': True,
                'layer_norm': False,
                'layers': {
                    'd': 256,
                    'n': 1
                },
                'nonlinearity': 'SiLU'
            },
        })
        m = rp3.load_model(config)

    
