import unittest
import ml_collections as mlc
import RP3Net as rp3
import tempfile
import gzip
import torch

_DUMMY_CLS_HEAD_CONFIG = {
    'embedding_dim': 1280,
    'bias': False,
    'end_bias': True,
    'layer_norm': False,
    'layers': {
        'd': 256,
        'n': 1
    },
    'nonlinearity': 'SiLU'
}

class RP3NetTest(unittest.TestCase):

    def test_fasta(self):
        f_name = tempfile.mktemp(suffix='.fasta')
        with open(f_name, 'w') as f:
            f.write('>seq#1.*_|foo=bar\nPRTEINWQENCE\nPRTEIN\nSQWENCE\n'
                    '>seq#2- baz\nACDEF\n\n'
                    '>этоя[]bak\nHJIK\n')
        seq_map = rp3.util.read_fasta(f_name)
        self.assertEqual(3, len(seq_map))
        self.assertEqual('PRTEINWQENCEPRTEINSQWENCE', seq_map['seq#1.*_'])
        self.assertEqual('ACDEF', seq_map['seq#2-'])
        self.assertEqual('HJIK', seq_map['этоя'])
        
        f_name = tempfile.mktemp(suffix='.fasta.gz')
        with gzip.open(f_name, 'wt') as f:
            f.write('>seq#1.*_|foo=bar\nPRTEINWQENCE\nPRTEIN\nSQWENCE\n'
                    '>seq#2- baz\nACDEF\n\n'
                    '>seq3[]bak\nHJIK\n\n')
        seq_map = rp3.util.read_fasta(f_name)
        self.assertEqual(3, len(seq_map))
        self.assertEqual('PRTEINWQENCEPRTEINSQWENCE', seq_map['seq#1.*_'])
        self.assertEqual('ACDEF', seq_map['seq#2-'])
        self.assertEqual('HJIK', seq_map['seq3'])
    
    def test_default_esm2_config_map(self):
        m = rp3.load_model(rp3.RP3_DEFAULT_CONFIG)
        score = m.predict(['PRTEINWQENCE', 'PRTEIN', 'SQWENCE'])
        self.assertTrue(isinstance(score, torch.Tensor))
        self.assertEqual(score.shape, (3,))
        self.assertTrue((score > 0).all())
        self.assertTrue((score < 1).all())
        score_map = m.predict({'seq1': 'PRTEINWQENCE', 'seq2': 'PRTEIN', 'seq3': 'SQWENCE'})
        self.assertTrue(isinstance(score_map, dict))
        self.assertEqual(len(score_map), 3)
        self.assertAlmostEqual(score_map['seq1'], score[0].item())
        self.assertAlmostEqual(score_map['seq2'], score[1].item())
        self.assertAlmostEqual(score_map['seq3'], score[2].item())
        
    def test_esm2_mean(self):
        config = mlc.FrozenConfigDict({
            'fm':{'type': 'esm2_650m'},
            'aggregation': 'mean',
            'classification_head': _DUMMY_CLS_HEAD_CONFIG,
        })
        m = rp3.load_model(config)

    def test_esm2_max(self):
        config = mlc.FrozenConfigDict({
            'fm':{'type': 'esm2_650m'},
            'aggregation': 'max',
            'classification_head': _DUMMY_CLS_HEAD_CONFIG,
        })
        m = rp3.load_model(config)
