import os
import typing
import functools
import zipfile

import torch
import torch.utils.data as torch_data
import polars as pl
import numpy as np
import lightning as L
import ml_collections as mlc

from .lm import RP3LM
from .. import util
from .. import model

log = util.get_logger(__name__)

FULL_DF_DTYPE_PL = {'created_at': pl.Datetime(), 'source': pl.Categorical(), 'sub_source': pl.Categorical(),
                 'no_tags_cluster_40_id': pl.String(), 'with_tags_cluster_90_id': pl.String(),
                 'has_dna': pl.Boolean(),
                 'experiment_id': pl.String(), 
                 'yield_binary': pl.Boolean(), 'yield_cat': pl.Int64(),
                 'host': pl.Categorical(), 'exp_outcome': pl.Categorical(), 
                 'id': pl.String(), 'fasta_id': pl.String(),  'dna_fasta_id': pl.String(), 'fasta_id_no_tags': pl.String(),
                 'ds_type': pl.Categorical(),
                 'n_tags_end': pl.Int64(), 'c_tags_start': pl.Int64(), 'n_fragments': pl.Int64(), 'unique_target_count': pl.Int64(),
                 'fasta_id_no_tags': pl.String(),
                 'uniprot_id': pl.String(), 'gene_id': pl.String(), 'taxon_id': pl.Int64(),
                 }

def read_full_df_pl(path:str|os.PathLike, **kwargs) -> pl.DataFrame:
    _schema = FULL_DF_DTYPE_PL if 'schema_overrides' not in kwargs else kwargs['schema_overrides']
    kwargs.pop('schema_overrides', None)
    df = pl.read_csv(path, schema_overrides=_schema, **kwargs)
    return df

def load_global_embeddings_file(embeddings_file:os.PathLike) -> typing.Mapping:
    log.info(f"Loading global embeddings from {embeddings_file}")
    embeddings_data = torch.load(embeddings_file)
    ids = embeddings_data['ids']
    embeddings = embeddings_data['embeddings']
    return {id: embeddings[i] for i, id in enumerate(ids)}

class RP3GlobalEmbeddingsDataSet(torch_data.Dataset):
    def __init__(self, df: pl.DataFrame, prefix:str, embeddings:typing.Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.df = df
        self.prefix=prefix
        self.embeddings = embeddings

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        try:
            row = self.df.row(idx, named=True)
            ret = {
                'idx': row['src_idx'], 
                'yield_binary': int(row['yield_binary']),
                'source': row['source'],
                'embeddings': self.embeddings[row['id']]
            }
            log.debug(f"{self.prefix}: (torch={idx}, csv={ret['idx']}){row['id']}: {ret['yield_binary']}")
            return ret
        except Exception as e:
            log.error(f"Top level catch in {self.prefix} __getitem__", exc_info=e)
            raise e


class RP3SequenceEmbeddingsDataSet(torch_data.Dataset):
    def __init__(self, df: pl.DataFrame, prefix:str, embeddings:zipfile.ZipFile) -> None:
        super().__init__()
        self.df = df
        self.prefix=prefix
        self.embeddings:zipfile.ZipFile = embeddings

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        try:
            row = self.df.row(idx, named=True)
            with self.embeddings.open(row['id'], 'r') as f:
                emb = torch.load(f, weights_only=True)
            ret = {
                'idx': row['src_idx'], 
                'yield_binary': int(row['yield_binary']),
                'source': row['source'],
                'embeddings': emb
            }
            log.debug(f"{self.prefix}: (torch={idx}, csv={ret['idx']}){row['id']}: {ret['yield_binary']}")
            return ret
        except Exception as e:
            log.error(f"Top level catch in {self.prefix} __getitem__", exc_info=e)
            raise e

class RP3SequenceDataSet(torch_data.Dataset):
    def __init__(self, df: pl.DataFrame, prefix, rng:np.random.Generator, max_seq_len:int=0):
        super().__init__()
        self.rng = rng
        self.df = df
        self.prefix=prefix
        self.max_seq_len = max_seq_len

    def seq_chunk(self, seq:str):
        if self.max_seq_len == 0 or len(seq) <= self.max_seq_len:
            log.debug(f"Not changing sequence of length {len(seq)}; max_seq_len={self.max_seq_len}")
            return seq
        start_idx = self.rng.integers(len(seq) - self.max_seq_len + 1)
        end_idx = start_idx + self.max_seq_len
        log.debug(f"Returning the {start_idx}:{end_idx} chunk from sequence of length {len(seq)}; max_seq_len={self.max_seq_len}")
        return seq[start_idx:end_idx]
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        try:
            row = self.df.row(idx, named=True)
            ret = {
                'idx': row['src_idx'],
                'source': row['source'],
                'seq': self.seq_chunk(row['seq']), 
                'yield_binary': int(row['yield_binary']),
            }
            log.debug(f"{self.prefix}: (torch={idx}, csv={ret['idx']}){row['id']}: {ret['yield_binary']}")
            return ret
        except Exception as e:
            log.error(f"Top level catch in {self.prefix} __getitem__", exc_info=e)
            raise e


class RP3LDM(L.LightningDataModule):
    def __init__(self, hypers) -> None:
        super().__init__()
        log.debug("DataModule init")
        self.save_hyperparameters({'data': hypers})
        self.hypers = mlc.ConfigDict(self.hparams.data)
        self.sources_map = {s: i for i, s in enumerate(self.hypers.sources)}
        self.rng = np.random.default_rng(self.hypers.get('seed', None))
        self.validation_slice = self.hypers.get('validation_slice', 'VALIDATION')
     
    def torch_dataset(self, df:pl.DataFrame, prefix:str) -> torch_data.Dataset:
        raise NotImplemented()

    def load_df(self) -> pl.DataFrame:
        data_path = self.hypers.ds_path
        log.info(f"Loading data from {data_path}; validation slice: {self.validation_slice}")
        df = read_full_df_pl(data_path).with_row_index('src_idx')
        df_sources = set(*df.select(pl.col('source').cast(pl.String).unique()))
        for s in self.sources_map:
            assert s in df_sources
        df = (df
            .filter(pl.col('source').is_in(self.hypers.sources))
            .with_columns(pl.col('source').cast(pl.String).replace_strict(self.sources_map))
        )
        return df
    
    def setup(self, stage: str) -> None:
        log.debug("RP3LDM setup")
        if self.trainer is not None:
            assert self.trainer.model.sources == self.hypers.sources
        df = self.load_df()
        self.df_train = df.filter(
            pl.col('ds_type').is_not_null() &
            pl.col('ds_type').is_in(['TEST', self.validation_slice]).not_()
        )
        assert self.df_train.shape[0] > 0, f"No training data for slice {self.validation_slice}"
        self.df_val = df.filter(ds_type=self.validation_slice)
        assert self.df_val.shape[0] > 0, f"No validation data for slice {self.validation_slice}"
        self.df_test = df.filter(ds_type='TEST')
        assert self.df_test.shape[0] > 0, f"No test data for slice {self.validation_slice}"
        self.create_torch_datasets()

    def create_torch_datasets(self):
        self.train_ds = self.torch_dataset(self.df_train, 'train')
        self.val_ds = self.torch_dataset(self.df_val, "val")
        df_val_train = self.df_train.sample(len(self.val_ds), with_replacement=False, seed=self.hypers.test_val_seed)
        self.val_train_ds = self.torch_dataset(df_val_train, "val-training")
        self.test_ds = self.torch_dataset(self.df_test, "test")

    def get_collate_fn(self):
        return None
    
    def get_batch_size(self, key:str) -> int:
        return int(self.hypers.get(f'{key}_batch_size', self.hypers.get('batch_size', -1)))

    def train_dataloader(self):
        batch_size = self.get_batch_size('training')
        return torch_data.DataLoader(self.train_ds, batch_size=batch_size, collate_fn=self.get_collate_fn(), shuffle=True,
                                     num_workers=0, pin_memory=True)
    
    def _build_val_test_loader(self, ds):
        batch_size = self.get_batch_size('val_test')
        sampler = None
        drop_last = False
        if self.hypers.get('use_distributed_sampler', False) and util.is_distr_env():
            sampler = torch_data.DistributedSampler(ds, drop_last=True, shuffle=False)
            drop_last = True
        dl = torch_data.DataLoader(ds, batch_size=batch_size, collate_fn=self.get_collate_fn(), sampler=sampler,
                                   num_workers=0, pin_memory=True, drop_last=drop_last)
        return dl

    def val_dataloader(self):
        train_dl = self._build_val_test_loader(self.val_train_ds)
        val_dl = self._build_val_test_loader(self.val_ds)
        return [train_dl, val_dl]
    
    def test_dataloader(self):
        return self._build_val_test_loader(self.test_ds)
    
class RP3GlobalEmbeddingsLDM(RP3LDM):
    def __init__(self, hypers) -> None:
        super().__init__(hypers)
        self.embeddings = None

    def load_df(self):
        df = super().load_df()
        embeddings_file = self.hypers.embeddings_file
        if embeddings_file == 'onehot' or embeddings_file.startswith('random_'):
            seqs = util.read_fasta(self.hypers.fasta_path)
            aa_to_int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
            if embeddings_file.startswith('random_'):
                emb = torch.nn.Embedding(20, int(embeddings_file[7:])).to('cpu').requires_grad_(False)
            self.embeddings = dict()
            for row in df.select('id', self.hypers.fasta_key).iter_rows():
                seq = seqs[row[1]]
                seq_tz = torch.tensor([aa_to_int[aa] for aa in seq], dtype=torch.int64)
                if embeddings_file == 'onehot':
                    seq_enc = torch.nn.functional.one_hot(seq_tz, num_classes=len(aa_to_int)).to(dtype=torch.float32)
                else:
                    seq_enc = emb(seq_tz)
                self.embeddings[row[0]] = seq_enc.mean(0)
        else:
            embeddings_file = util.resolve(self.hypers.embeddings_file)
            self.embeddings = load_global_embeddings_file(embeddings_file)
        return df

    def torch_dataset(self, df:pl.DataFrame, prefix:str) -> torch_data.Dataset:
        return RP3GlobalEmbeddingsDataSet(df, prefix, self.embeddings)
    
class RP3SequenceEmbeddingsLDM(RP3LDM):
    def __init__(self, hypers) -> None:
        super().__init__(hypers)
        self.embeddings_file = None

    def load_df(self):
        df = super().load_df()
        self.embeddings_file = zipfile.ZipFile(util.resolve(self.hypers.embeddings_file), 'r')
        return df
    
    @staticmethod
    def collate(batch):
        embeddings = [b.pop('embeddings') for b in batch]
        ret = torch_data.default_collate(batch)
        emb_len = torch.tensor([e.shape[0] for e in embeddings])
        max_len = emb_len.max()
        emb_padded = torch.stack([torch.nn.functional.pad(e, (0,0,0, max_len - e.shape[0]), value=0) for e in embeddings])
        attn_mask = torch.zeros((emb_len.shape[0], max_len), dtype=torch.int32)
        for i, l in enumerate(emb_len):
            attn_mask[i, :l] = 1
        ret['embeddings'] = emb_padded
        ret['attention_mask'] = attn_mask
        return ret
    
    def get_collate_fn(self):
        return RP3SequenceEmbeddingsLDM.collate
    
    def torch_dataset(self, df:pl.DataFrame, prefix:str) -> torch_data.Dataset:
        return RP3SequenceEmbeddingsDataSet(df, prefix, self.embeddings_file)
    

class RP3SequenceLDM(RP3LDM):

    def __init__(self, hypers) -> None:
        super().__init__(hypers)

    def load_df(self):
        df = super().load_df()
        log.info(f"Reading sequences from {self.hypers.fasta_path}")
        fasta_map = util.read_fasta(self.hypers.fasta_path)
        fasta_id_col = self.hypers.fasta_id_col
        df = df.with_columns(seq=pl.col(fasta_id_col).replace_strict(fasta_map))
        return df
    
    def torch_dataset(self, df:pl.DataFrame, prefix:str) -> torch_data.Dataset:
        return RP3SequenceDataSet(df, prefix, self.rng, self.hypers.get('max_seq_len', 0))

    @staticmethod
    def collate(tokenizer:model.RP3Net, batch):
        seqs = [b.pop('seq') for b in batch]
        ret = torch_data.default_collate(batch)
        ret['seq'] = tokenizer.tokenize_sequences(seqs)
        return ret
    
    def get_collate_fn(self):
        lm: RP3LM = self.trainer.lightning_module
        return functools.partial(RP3SequenceLDM.collate, lm.model)
    