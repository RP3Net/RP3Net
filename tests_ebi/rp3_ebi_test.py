import unittest
import tempfile
import RP3Net as rp3
import ml_collections as mlc


class RP3NetEbiTest(unittest.TestCase):
    def test_training_a(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_a.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])

    def test_training_b(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_b.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])

    def test_training_d(self):
        rootdir = tempfile.mkdtemp(dir=rp3.util.util.resolve('.'))
        rp3.training.cli.RP3Cli(args=['fit', '-c', './config/trainer_d.yml', '-c', './config/trainer_ebi_test.yml', '--trainer.default_root_dir', rootdir])

    def test_ebi_az_1(self):
        m = rp3.load_model(rp3.RP3_CONFIG_B, '/homes/evgeny/data/protman_checkpoints/protman_b_az_1.ckpt')
        score = m.predict(['PRTEINWQENCE', 'PRTEIN', 'SQWENCE'])
        self.assertEqual(score.shape, (3,))
        self.assertTrue((score > 0).all())
        self.assertTrue((score < 1).all())

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

    
