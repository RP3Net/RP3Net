from typing import Mapping, List, Tuple
import torchmetrics as tmx
import torch
from .. import util

import logging
log = util.get_logger(__name__)
# log.setLevel(logging.DEBUG)


_MULTICLASS_MX_CLS = (tmx.classification.MulticlassAUROC |
                      tmx.classification.MulticlassAveragePrecision |
                      tmx.classification.MulticlassAccuracy |
                      tmx.classification.MulticlassPrecisionAtFixedRecall |
                      tmx.classification.MulticlassMatthewsCorrCoef |
                      tmx.classification.MulticlassF1Score)
_BINARY_MX_CLS = (tmx.classification.BinaryPrecisionAtFixedRecall)

class ClassificationMetricContainer(torch.nn.Module):
    """
    This class is a wrapper around torchmetrics.Metric instances that allows for easy handling of multiple sources.

    Make sure logits are softmaxed in here before passing them on to torch classification metrics.
    https://github.com/Lightning-AI/torchmetrics/issues/1604
    https://github.com/Lightning-AI/torchmetrics/issues/2195
    https://github.com/Lightning-AI/torchmetrics/pull/1676

    In order for checkpointing to work properly, the monitored keys have to be logged as metrics, not as values. 
    This in turn requires this class to be a `torch.nn.Module`, and the monitored metrics to be stored as submodules inside a `torch.nn.ModuleDict`.
    This is also needed for the `to` method and ddp to work properly.

    We update the metrics with dummy values if update has not been called before compute, to avoid errors.
    """

    @staticmethod
    def create_classification_metrics(sources: list[str], n_classes:int):
        return ClassificationMetricContainer(
            train_metrics={
                'acc': tmx.classification.BinaryAccuracy(),
                'auc_roc': tmx.classification.MulticlassAUROC(n_classes, thresholds=100),
                'acc': tmx.classification.MulticlassAccuracy(n_classes),
                'mcc' : tmx.classification.MulticlassMatthewsCorrCoef(n_classes),
                'f1' : tmx.classification.MulticlassF1Score(n_classes),
                'bpfr_090': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.9, thresholds=100),
                'bpfr_075': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.75, thresholds=100),
                'bpfr_050': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.5, thresholds=100),
                'bpfr_025': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.25, thresholds=100),
            },
            val_metrics={
                'auc_roc': tmx.classification.MulticlassAUROC(n_classes, thresholds=100),
                'acc': tmx.classification.MulticlassAccuracy(n_classes),
                'mcc' : tmx.classification.MulticlassMatthewsCorrCoef(n_classes),
                'f1' : tmx.classification.MulticlassF1Score(n_classes),
                'bpfr_090': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.9, thresholds=100),
                'bpfr_075': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.75, thresholds=100),
                'bpfr_050': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.5, thresholds=100),
                'bpfr_025': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.25, thresholds=100),
            },
            test_metrics={
                'auc_roc': tmx.classification.MulticlassAUROC(n_classes, thresholds=100),
                'acc': tmx.classification.MulticlassAccuracy(n_classes),
                'mcc' : tmx.classification.MulticlassMatthewsCorrCoef(n_classes),
                'f1' : tmx.classification.MulticlassF1Score(n_classes),
                'bpfr_090': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.9, thresholds=100),
                'bpfr_075': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.75, thresholds=100),
                'bpfr_050': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.5, thresholds=100),
                'bpfr_025': tmx.classification.MulticlassPrecisionAtFixedRecall(n_classes, min_recall=0.25, thresholds=100),
            },
            checkpointing_metric_key='auc_roc',
            y_batch_key='yield_binary',
            idx_batch_key='idx',
            source_key='source',
            sources=sources
        )
    
    @staticmethod
    def create_empty_metrics():
        return ClassificationMetricContainer(
            train_metrics={},
            val_metrics={},
            test_metrics={},
            y_batch_key='yield_binary',
            idx_batch_key='idx',
        )

    def __init__(self, 
                 train_metrics:Mapping[str, tmx.Metric], 
                 val_metrics:Mapping[str, tmx.Metric], 
                 test_metrics:Mapping[str, tmx.Metric],
                 y_batch_key:str,
                 idx_batch_key:str,
                 checkpointing_metric_key:str|None=None,
                 source_key:str|None=None,
                 sources:List[str]|None=None,
                 ys_dtype:torch.dtype|None=None,
                 key_separator:str='_') -> None:
        super().__init__()
        def make_key(*args):
            return key_separator.join(args)
        self.ys_dtype = ys_dtype
        self.checkpointing_metrics = torch.nn.ModuleDict()
        self.train_metrics = torch.nn.ModuleDict({make_key('train', k):m for k, m in train_metrics.items()})
        if checkpointing_metric_key is not None:
            self.checkpointing_metrics[make_key('train', checkpointing_metric_key)] = train_metrics[checkpointing_metric_key]
        self.sources = sources
        if sources is None:
            self.val_metrics = torch.nn.ModuleDict({make_key('val', k):m for k, m in val_metrics.items()})
            if checkpointing_metric_key is not None:
                self.checkpointing_metrics[make_key('val', checkpointing_metric_key)] = val_metrics[checkpointing_metric_key]
            self.test_metrics = torch.nn.ModuleDict({make_key('test', k):m for k, m in test_metrics.items()})
        else:
            for s in sources:
                assert s != 'full', 'Cannot use "full" as a source key, it is reserved for the full dataset.'
            self.source_key = source_key
            self.val_metrics = torch.nn.ModuleDict()
            self.val_metrics['full'] = torch.nn.ModuleDict({make_key('val', k):m for k, m in val_metrics.items()})
            if checkpointing_metric_key is not None:
                self.checkpointing_metrics[make_key('full', 'val', checkpointing_metric_key)] = val_metrics[checkpointing_metric_key]
            for source in sources:
                tmp = torch.nn.ModuleDict()
                for k, m in val_metrics.items():
                    m = m.clone()
                    tmp[make_key('val', k)] = m
                    if k == checkpointing_metric_key:
                        self.checkpointing_metrics[make_key(source, 'val', checkpointing_metric_key)] = m
                self.val_metrics[source] = tmp
            self.test_metrics = torch.nn.ModuleDict({source: torch.nn.ModuleDict({make_key('test', k):m.clone() for k,m in test_metrics.items()}) for source in sources})
            self.test_metrics['full'] = torch.nn.ModuleDict({make_key('test', k):m for k, m in test_metrics.items()})
        self.train_ids = tmx.aggregation.CatMetric('error')
        self.val_ids = tmx.aggregation.CatMetric('error')
        self.train_logits = tmx.aggregation.CatMetric('error')
        self.val_logits = tmx.aggregation.CatMetric('error')
        self.test_ids = tmx.aggregation.CatMetric('error')
        self.test_logits = tmx.aggregation.CatMetric('error')
        self.key_separator = key_separator
        self.y_batch_key = y_batch_key
        self.idx_batch_key = idx_batch_key
        self.sentinel = torch.nn.Parameter(torch.zeros(1))

    
    def apply_rec(self, d:Mapping|tmx.Metric, f):
        if isinstance(d, Mapping|torch.nn.ModuleDict):
            return {k: self.apply_rec(v, f) for k, v in d.items()}
        else:
            return f(d)
    
    def flatten(self, d:Mapping|tmx.Metric, f=lambda x, k: x, in_key=''):
        if isinstance(d, Mapping|torch.nn.ModuleDict):
            ret = {}
            for k, v in d.items():
                key = in_key
                if k != '':
                    if len(key) > 0:
                        key += self.key_separator
                    key += k
                ret |= self.flatten(v, f, key)
            return ret
        else:
            log.debug(f"calling function from flatten {in_key}")
            fd = f(d, in_key)
            if isinstance(fd, Mapping|torch.nn.ModuleDict):
                return {in_key + self.key_separator + k:v for k, v in fd.items()}
            elif fd is None:
                return {}
            else:
                return {in_key: fd}
    
    def update_by_source(self, metrics, logits, batch):
        ys = batch[self.y_batch_key]
        if self.ys_dtype is not None:
            ys = ys.to(dtype=self.ys_dtype)
        if self.sources is not None:
            for m in metrics['full'].values():
                log.debug(f"Updating full metric {m} with logits {logits} and ys {ys}")
                self.update_metric(m, logits, ys)
            batch_source = batch[self.source_key]
            for i, source in enumerate(self.sources):
                if (batch_source == i).sum() > 0:
                    _logits = logits[batch_source == i]
                    _ys = ys[batch_source == i]
                    for m in metrics[source].values():
                        log.debug(f"Updating {source} metric {m} with logits {logits} and ys {ys}")
                        self.update_metric(m, _logits, _ys)
        else:
            for m in metrics.values():
                log.debug(f"Updating full metric {m} with logits {logits} and ys {ys}")
                self.update_metric(m, logits, ys)

    @staticmethod
    def update_multiclass(m, logits, ys):
        probs = torch.softmax(logits, dim=-1) # see class docstring
        m.update(probs, ys)

    @staticmethod
    def update_binary(m, logits, ys):
        assert logits.shape[-1] == 2
        probs = torch.sigmoid(logits.diff(dim=-1).squeeze(-1)) # see class docstring
        m.update(probs, ys)

    @classmethod
    def update_metric(cls, m, logits, ys):
        if isinstance(m, _MULTICLASS_MX_CLS):
            cls.update_multiclass(m, logits, ys)
        elif isinstance(m, _BINARY_MX_CLS):
            cls.update_binary(m, logits, ys)
        else:
            raise Exception(f"Unsupported metric type {type(m)}")
        
    def update_sentinel_multiclass(self, m):
        logits = torch.zeros((1, m.num_classes), dtype=torch.float, device=self.sentinel.device) * float('nan')
        ys = torch.zeros(1, dtype=torch.int64, device=self.sentinel.device)
        m.update(logits, ys)

    def update_sentinel_binary(self, m):
        logits = torch.zeros(1, dtype=torch.float, device=self.sentinel.device) * float('nan')
        ys = torch.zeros(1, dtype=torch.int64, device=self.sentinel.device)
        m.update(logits, ys)

    def update_sentinel(self, m):
        if isinstance(m, _MULTICLASS_MX_CLS):
            self.update_sentinel_multiclass(m)
        elif isinstance(m, _BINARY_MX_CLS):
            self.update_sentinel_binary(m)
        else:
            raise Exception(f"Unsupported metric type {type(m)}")

    @staticmethod
    def log_summary(vv):
        if log.isEnabledFor(logging.DEBUG):
            return [(v.shape, v.device) for v in vv]
        else:
            return ''

    def update_train(self, logits, batch):
        ys = batch[self.y_batch_key]
        if self.ys_dtype is not None:
            ys = ys.to(dtype=self.ys_dtype)
        for m in self.train_metrics.values():
            self.update_metric(m, logits, ys)
        ids = batch[self.idx_batch_key]
        self.train_ids.update(ids)
        log.debug(f"Train ids after update: {self.log_summary(self.train_ids.value)}")
        self.train_logits.update(logits)
        log.debug(f"Train logits after update: {self.log_summary(self.train_logits.value)}")

    def update_val(self, logits, batch):
        self.update_by_source(self.val_metrics, logits, batch)
        ids = batch[self.idx_batch_key]
        self.val_ids.update(ids)
        log.debug(f"Val ids after update: {self.log_summary(self.val_ids.value)}")
        self.val_logits.update(logits)
        log.debug(f"Val logits after update: {self.log_summary(self.val_logits.value)}")

    def update_test(self, logits, batch):
        self.update_by_source(self.test_metrics, logits, batch)
        ids = batch[self.idx_batch_key]
        self.test_ids.update(ids)
        log.debug(f"Test ids after update: {self.log_summary(self.test_ids.value)}")
        self.test_logits.update(logits)
        log.debug(f"Test logits after update: {self.log_summary(self.test_logits.value)}")


    def compute_specialized(self, m:tmx.Metric, key:str):
        if not m.update_called:
            log.debug(f"Calling update on {key}:{m} to avoid unaligned distributed calls across workers")
            self.update_sentinel(m)
        if m in set(self.checkpointing_metrics.values()):            
            return m
        if m.update_called:
            tz = m.compute()
            if isinstance(tz, torch.Tensor):
                if tz.numel() == 1:
                    return tz.item()
                elif isinstance(m, (tmx.classification.BinaryConfusionMatrix, 
                                    tmx.classification.MulticlassConfusionMatrix)) \
                        and tz.shape == (2,2):
                    tp, tn, fp, fn = tz.flatten()
                    all = tz.sum()
                    return {
                        'tp': (tp/all).item(),
                        'tn': (tn/all).item(),
                        'fp': (fp/all).item(),
                        'fn': (fn/all).item(),
                        'accuracy': ((tp + tn) / (tp + tn + fp + fn)).item(),
                        'precision': (tp / (tp + fp)).item(),
                        'recall': (tp / (tp + fn)).item(),
                        'f1': (2 * tp / (2 * tp + fp + fn)).item(),
                        'mcc': ((tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5).item()
                    }
                else:
                    raise Exception(f"Unsupported shape {tz.shape} for {key}")
            elif isinstance(m, tmx.classification.MulticlassPrecisionAtFixedRecall):  # PrecisionAtFixedRecall returns (precision, threshold), each of shape (n_classes,)
                return tz[0].mean().item()
            else: 
                raise Exception(f"Unsupported return type {type(m)}:{type(tz)} for {key}")
        else:
            log.warning(f"Update not called for {key}:{m} - this should never happen")
            return float("nan")

    def compute_train_dict(self) -> Mapping[str, float]:
        return self.flatten(self.train_metrics, self.compute_specialized)

    def compute_val_dict(self) -> Mapping[str, float]:
        return self.flatten(self.val_metrics, self.compute_specialized)

    def compute_test_dict(self) -> Mapping[str, float]:
        return self.flatten(self.test_metrics, self.compute_specialized)

    def train_curve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        log.debug(f"Train ids before compute: {self.log_summary(self.train_ids.value)}")
        ids = self.train_ids.compute()
        log.debug(f"Train ids after compute: {self.log_summary(self.train_ids.value)}")
        log.debug(f"Train logits before compute: {self.log_summary(self.train_logits.value)}")
        logits = self.train_logits.compute()
        log.debug(f"Train logits after compute: {self.log_summary(self.train_logits.value)}")
        return ids, logits

    def val_curve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        log.debug(f"Val ids before compute: {self.log_summary(self.val_ids.value)}")
        ids = self.val_ids.compute()
        log.debug(f"Val ids after compute: {self.log_summary(self.val_ids.value)}")
        log.debug(f"Val logits before compute: {self.log_summary(self.val_logits.value)}")
        logits = self.val_logits.compute()
        log.debug(f"Val logits after compute: {self.log_summary(self.val_logits.value)}")
        return ids, logits

    def test_curve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.test_ids.compute()
        logits = self.test_logits.compute()
        return ids, logits

    def flat_train(self):
        return self.flatten(self.train_metrics)
    
    def flat_val(self):
        return self.flatten(self.val_metrics)
    
    def flat_test(self):
        return self.flatten(self.test_metrics)
    
    def reset(self):
        def _reset(x):
            if x not in set(self.checkpointing_metrics.values()):
                x.reset()
            return x
        self.apply_rec(self.train_metrics, _reset)
        self.apply_rec(self.val_metrics, _reset)
        self.apply_rec(self.test_metrics, _reset)
        self.train_ids.reset()
        self.val_ids.reset()
        self.train_logits.reset()
        self.val_logits.reset()
        self.test_ids.reset()
        self.test_logits.reset()

    def metrics_for_checkpointing(self):
        return self.checkpointing_metrics
