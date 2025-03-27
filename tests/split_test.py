import unittest
import torch
import RP3Net as rp3

class BatchSplitTest(unittest.TestCase):
    def test_split_key_index(self):
        ix = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        b = {
            'ix': ix,
            'foo': {'bar': torch.rand(9, 5)},
            'baz': [f"str{i}" for i in range(9)]
        }
        b1, b2 = rp3.util.torch_split_key_index(b, 'ix', torch.tensor([0, 2]))
        self.assertEqual(b1['ix'].tolist(), [0, 0, 0, 2, 2, 2])
        self.assertEqual(b2['ix'].tolist(), [1, 1, 1])
        self.assertTrue(torch.equal(b1['foo']['bar'], b['foo']['bar'][[0,1,2,6,7,8],:]))
        self.assertTrue(torch.equal(b2['foo']['bar'], b['foo']['bar'][[3,4,5], :]))
        self.assertEqual(b1['baz'], [f"str{i}" for i in range(3)] + [f"str{i}" for i in range(6, 9)])
        self.assertEqual(b2['baz'], [f"str{i}" for i in range(3, 6)])

    def test_split_key_index_empty_1(self):
        ix = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        b = {
            'ix': ix,
            'foo': {'bar': torch.rand(9, 5)},
            'baz': [f"str{i}" for i in range(9)]
        }
        b1, b2 = rp3.util.torch_split_key_index(b, 'ix', torch.tensor([4,5]))
        self.assertEqual(b1['ix'].tolist(), [])
        self.assertEqual(b2['ix'].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(torch.equal(b1['foo']['bar'], torch.empty((0,5))))
        self.assertTrue(torch.equal(b2['foo']['bar'], b['foo']['bar']))
        self.assertEqual(b1['baz'], [])
        self.assertEqual(b2['baz'], b['baz'])

    def test_split_key_index_empty_2(self):
        ix = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        b = {
            'ix': ix,
            'foo': {'bar': torch.rand(9, 5)},
            'baz': [f"str{i}" for i in range(9)]
        }
        b1, b2 = rp3.util.torch_split_key_index(b, 'ix', torch.tensor([0,1,2]))
        self.assertEqual(b1['ix'].tolist(), [0, 0, 0, 1,1,1,2, 2, 2])
        self.assertEqual(b2['ix'].tolist(), [])
        self.assertTrue(torch.equal(b1['foo']['bar'], b['foo']['bar']))
        self.assertTrue(torch.equal(b2['foo']['bar'], torch.empty((0,5))))
        self.assertEqual(b1['baz'], b['baz'])
        self.assertEqual(b2['baz'], [])

    def test_split_size(self):
        ix = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        b = {
            'ix': ix,
            'foo': {'bar': torch.rand(9, 5)},
            'baz': [f"str{i}" for i in range(9)]
        }
        b1,b2,b3 = rp3.util.torch_split_size(b, (2, 3, 4))
        self.assertEqual(b1['ix'].tolist(), [0, 0])
        self.assertEqual(b2['ix'].tolist(), [0,1,1])
        self.assertEqual(b3['ix'].tolist(), [1,2,2,2])
        self.assertTrue(torch.equal(b1['foo']['bar'], b['foo']['bar'][:2,:]))
        self.assertTrue(torch.equal(b2['foo']['bar'], b['foo']['bar'][2:5,:]))
        self.assertTrue(torch.equal(b3['foo']['bar'], b['foo']['bar'][5:, :]))
        self.assertEqual(b1['baz'], [f"str{i}" for i in range(2)] )
        self.assertEqual(b2['baz'], [f"str{i}" for i in range(2, 5)])
        self.assertEqual(b3['baz'], [f"str{i}" for i in range(5, 9)])

    def test_split_chunks(self):
        ix = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        b = {
            'ix': ix,
            'foo': {'bar': torch.rand(9, 5)},
            'baz': [f"str{i}" for i in range(9)]
        }
        chunks = rp3.util.torch_split_chunks(b, 4)
        chunks = list(chunks)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]['ix'].tolist(), [0, 0, 0])
        self.assertEqual(chunks[1]['ix'].tolist(), [1, 1])
        self.assertEqual(chunks[2]['ix'].tolist(), [1, 2])
        self.assertEqual(chunks[3]['ix'].tolist(), [2, 2])
        self.assertTrue(torch.equal(chunks[0]['foo']['bar'], b['foo']['bar'][:3,:]))
        self.assertTrue(torch.equal(chunks[1]['foo']['bar'], b['foo']['bar'][3:5,:]))
        self.assertTrue(torch.equal(chunks[2]['foo']['bar'], b['foo']['bar'][5:7,:]))
        self.assertTrue(torch.equal(chunks[3]['foo']['bar'], b['foo']['bar'][7:9,:]))
        self.assertEqual(chunks[0]['baz'], [f"str{i}" for i in range(3)])
        self.assertEqual(chunks[1]['baz'], [f"str{i}" for i in range(3,5)])
        self.assertEqual(chunks[2]['baz'], [f"str{i}" for i in range(5,7)])
        self.assertEqual(chunks[3]['baz'], [f"str{i}" for i in range(7,9)])

    def test_concat(self):
        b1 = {
            'ix': torch.arange(3),
            'foo': {'bar': torch.rand(3, 5)},
            'baz': [f"str{i}" for i in range(3)]
        }
        b2 = {
            'ix': torch.arange(6) + 3,
            'foo': {'bar': torch.rand(6, 5) + 100},
            'baz': [f"foo{i}" for i in range(6)]
        }
        b = rp3.util.torch_concat_chunks([b1, b2])
        self.assertTrue(torch.equal(b['ix'], torch.arange(9)))
        self.assertEqual(b['baz'], [f"str{i}" for i in range(3)] + [f"foo{i}" for i in range(6)])
        self.assertTrue(torch.equal(b['foo']['bar'], torch.cat([b1['foo']['bar'], b2['foo']['bar']], dim=0)))
