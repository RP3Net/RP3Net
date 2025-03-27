import logging
import numpy as np
import polars as pl
import torch.utils.data as torch_data

from . import data
from .. import util

log = util.get_logger(__name__)

class EmlcBatchSampler(torch_data.Sampler):
    def __init__(self, *, df:pl.DataFrame, rng:np.random.Generator, clean_sources:list[int], noisy_sources:list[int],
                  batch_size_clean:int, emlc_k:int=1, world_size:int=1, global_rank:int=0):
        self.df = df.with_row_index('_row_idx')
        self.rng = rng
        self.emlc_k = emlc_k
        self.clean_sources = clean_sources
        self.noisy_sources = noisy_sources
        self.batch_size_clean = batch_size_clean
        self.batch_size_noisy = batch_size_clean * emlc_k
        self.batch_count = min(df.select(pl.col('source').is_in(clean_sources)).sum()[0,0] // (self.batch_size_clean * world_size),
                               df.select(pl.col.source.is_in(noisy_sources)).sum()[0,0] // (self.batch_size_noisy * world_size))
        self.global_rank = global_rank
        self.world_size = world_size
        
    def __iter__(self):
        ix_clean = self.df.filter(pl.col.source.is_in(self.clean_sources)).select('_row_idx').to_numpy().flatten()
        self.rng.shuffle(ix_clean)
        ix_noisy = self.df.filter(pl.col.source.is_in(self.noisy_sources)).select('_row_idx').to_numpy().flatten()
        self.rng.shuffle(ix_noisy)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Clean index:\n{ix_clean[:100]}")
            log.debug(f"Noisy index:\n{ix_noisy[:100]}")
        ix_clean = ix_clean[self.global_rank::self.world_size]
        ix_noisy = ix_noisy[self.global_rank::self.world_size]
        log.debug(f"{self.batch_count} batches per worker {self.global_rank}/{self.world_size}")
        for i in range(self.batch_count):
            out = np.concatenate([ix_clean[i*self.batch_size_clean:(i+1)*self.batch_size_clean],
                                  ix_noisy[i*self.batch_size_noisy:(i+1)*self.batch_size_noisy]])
            log.debug(f"Batch {i}/{self.batch_count}: {out}")
            yield out

    def __len__(self):
        return self.batch_count
    

class EmlcLDM(data.RP3SequenceLDM):
    def __init__(self, hypers) -> None:
        super().__init__(hypers)
        log.info("EmlcLDM init")
        clean_sources = self.hypers.clean_sources
        self.emlc_k = int(self.hypers.emlc_k)
        self.clean_sources = [self.sources_map[s] for s in clean_sources]

    def setup(self, stage):
        super().setup(stage)
        if self.trainer is not None:
            assert self.trainer.model.clean_sources == self.clean_sources
            assert self.trainer.model.emlc_k == self.emlc_k

        
    def train_dataloader(self):
        log.debug("EmlcCVLDM train_dataloader")
        batch_size = self.get_batch_size('training')
        noisy_sources = [self.sources_map[s] for s in self.sources_map if self.sources_map[s] not in self.clean_sources]
        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank
        batch_sampler = EmlcBatchSampler(df=self.df_train, rng=self.rng, 
                                         clean_sources=self.clean_sources, noisy_sources=noisy_sources, 
                                         batch_size_clean=batch_size, emlc_k=self.emlc_k,
                                         world_size=world_size, global_rank=global_rank)
        return torch_data.DataLoader(self.train_ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, collate_fn=self.get_collate_fn())
    