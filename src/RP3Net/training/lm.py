import lightning as L
import lightning.pytorch.utilities as L_util

import torch.nn as nn
import torch
import ml_collections as mlc
import pandas as pd
import os

from . import metrics
from .. import util
from .. import model


log = util.get_logger(__name__)

class RP3LM(L.LightningModule):
    def __init__(self, hypers) -> None:
        super().__init__()
        log.debug("Lightning module init")
        self._hypers_prefix = 'model'
        self.save_hyperparameters({'model': hypers})
        self.hypers = mlc.ConfigDict(self.hparams.model)
        self.sources = self.hypers.sources
        self.sources_map = {s:i for i, s in enumerate(self.sources)}
        log.info(f"Sources: {self.sources}")
        self.metrics = metrics.ClassificationMetricContainer.create_classification_metrics(self.sources, 2)
        self.loss = nn.CrossEntropyLoss()
        log.info(f"Loss: {self.loss}")
        self.model: model.RP3Net = model.load_model(self.hypers.model)
        log.info(f"Model: {self.model}")

    def setup(self, stage):
        if stage == 'fit':
            assert self.model.mode in model.Mode_Training, "Model must be in training mode"

    def force_train_on_fit_start(self):
        """
        Need this, because loading a pre-trained HF model calls .eval() under the hood,
        and PL preserves the state of training flags on modules when switching back from eval to train.
        """
        self.model.train()

    def on_fit_start(self) -> None:
        self.force_train_on_fit_start()
    
    def forward(self, batch):
        return self.model(batch)
        
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self.model(batch)
        return torch.argmax(logits, dim=1)
    
    def training_step(self, batch, batch_idx):
        log.debug(f"Training batch ids: {batch['idx']}")        
        logits = self(batch)
        loss = self.loss(logits, batch['yield_binary'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch['yield_binary'].shape[0])
        return loss 

    def validation_step(self, batch, batch_idx, dataloader_idx):
        ids = batch['idx']
        log.debug(f"Validation batch ids for dataloader index {dataloader_idx}: {ids}")
        logits = self(batch)
        if dataloader_idx == 0:
            self.metrics.update_train(logits, batch)
        elif dataloader_idx == 1:
            self.metrics.update_val(logits, batch)
        else:
            raise RuntimeError(f"Unknown dataloader index: {dataloader_idx}")

    def test_step(self, batch, batch_idx):
        log.debug(f"Test batch index: {batch['idx']}")
        logits = self(batch)
        self.metrics.update_test(logits, batch)

    @L_util.rank_zero_only
    def write_results_df(self, filename:os.PathLike, ids:torch.Tensor, logits:torch.Tensor):
        proba = torch.softmax(logits, axis=1).cpu().numpy()
        y_hat = proba.argmax(axis=1)
        df = pd.DataFrame({'id': ids.to(dtype=torch.int32, device='cpu').numpy(), 'y_hat': y_hat})
        df_logits = pd.DataFrame(logits.cpu().numpy(), columns=[f'logit_{i}' for i in range(logits.shape[1])])
        df_proba = pd.DataFrame(proba, columns=[f'prob_{i}' for i in range(proba.shape[1])])
        df = pd.concat([df, df_logits, df_proba], axis=1)
        df.to_csv(filename, index=False)

    def on_validation_epoch_end(self) -> None:
        log.info(f"Validation epoch {self.current_epoch} end.")
        train_log_dict = self.metrics.compute_train_dict()
        if not self.trainer.sanity_checking:
            self.log_dict(train_log_dict, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        
        val_log_dict = self.metrics.compute_val_dict()
        if not self.trainer.sanity_checking:
            self.log_dict(val_log_dict, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        
        train_df_file = util.resolve(self.trainer.default_root_dir) / f"train_df_{self.current_epoch}.csv.gz"
        train_ids, train_logits = self.metrics.train_curve()
        if isinstance(train_logits, torch.Tensor) and train_logits.shape[0] > 0 and not self.trainer.sanity_checking:
            log.info(f"Writing training results for epoch {self.current_epoch} to {train_df_file}")
            self.write_results_df(train_df_file, train_ids, train_logits)
        
        val_df_file = util.resolve(self.trainer.default_root_dir) / f"val_df_{self.current_epoch}.csv.gz"
        val_ids, val_logits = self.metrics.val_curve()
        if isinstance(val_logits, torch.Tensor) and val_logits.shape[0] > 0 and not self.trainer.sanity_checking:
            log.info(f"Writing validation results for epoch {self.current_epoch} to {val_df_file}")
            self.write_results_df(val_df_file, val_ids, val_logits)
        self.metrics.reset()
        

    def on_test_epoch_end(self) -> None:
        test_log_dict = self.metrics.compute_test_dict()
        self.log_dict(test_log_dict, on_epoch=True, add_dataloader_idx=False)
        test_df_file = util.resolve(self.trainer.default_root_dir) / f"test_df.csv.gz"
        test_ids, test_logits = self.metrics.test_curve()
        if isinstance(test_logits, torch.Tensor) and test_logits.shape[0] > 0:
            log.info(f"Writing test results to {test_df_file}")
            self.write_results_df(test_df_file, test_ids, test_logits)
        self.metrics.reset()



                 