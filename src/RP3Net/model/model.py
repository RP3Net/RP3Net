import abc
import re
import os
import enum
import typing
import torch
import torch.nn as nn
import ml_collections as mlc
import transformers as hub
import pathlib
import peft

from . import layers
from .. import util

log = util.get_logger(__name__)

def resolve(filename) -> pathlib.Path:
    if type(filename) == str:
        filename = pathlib.Path(filename)
    if str(filename).startswith('~'):
        filename = filename.expanduser()
    return filename.resolve()

class Mode(enum.Flag):
    Inference = enum.auto()
    Training_A = enum.auto()
    Training_B = enum.auto()
    Training_C = enum.auto()
    Training_D = enum.auto()
Mode_Training = Mode.Training_A | Mode.Training_B | Mode.Training_C | Mode.Training_D
Mode_Training_FM = Mode.Training_C | Mode.Training_D
Mode_FM = Mode_Training_FM | Mode.Inference
Mode_Training_Aggregation = Mode.Training_B | Mode.Training_C | Mode.Training_D
Mode_Aggregation = Mode_Training_Aggregation | Mode.Inference

class RP3Net(nn.Module):
    def __init__(self, cfg:mlc.ConfigDict) -> None:
        super(RP3Net, self).__init__()
        self.fm = None
        self.mode = Mode[cfg.get('mode', 'Inference')]
        if self.mode in Mode_FM:
            self._init_fm(cfg, self.mode)
            assert self.fm is not None, "Model must be initialized"
        if self.mode in Mode_Aggregation:
            if cfg.aggregation == 'mean':
                self.pooling = layers.MeanPooling()
            elif cfg.aggregation == 'max':
                self.pooling = layers.MaxPooling()
            elif cfg.aggregation == 'stp':
                self.pooling = layers.SetTransformerPooling(**cfg.stp)
            else:
                raise ValueError(f"Aggregation type {cfg.aggregation} not supported")
        self.cls_head = layers.ClsHead(**cfg.classification_head)

    def forward(self, batch, return_repr=False):
        if self.mode == Mode.Training_A:
            logits = self.cls_head(batch['embeddings'])
        elif self.mode == Mode.Training_B:
            global_repr = self.pooling(batch['embeddings'], mask=batch['attention_mask'])
            logits = self.cls_head(global_repr)
        else: # Inference, Training_CD
            seq_repr = self.sequence_representation(batch['seq'])
            mask = self.attention_mask(batch['seq'])
            global_repr = self.pooling(seq_repr, mask=mask)
            logits = self.cls_head(global_repr)
        if return_repr:
            return logits, global_repr
        return logits
    
    # def train(self, train_mode:bool=True):
    #     if not train_mode or self.mode == Mode.Training_CD:
    #         super().train(train_mode)
    #     else:
    #         if self.mode == Mode.Inference:
    #             raise ValueError("Model is in inference mode")
    #         elif self.mode == Mode.Training_A:
    #             self.fm.train(False)
    #             self.pooling.train(False)
    #             self.cls_head.train(True)
    #         elif self.mode == Mode.Training_B:
    #             self.fm.train(False)
    #             self.pooling.train(True)
    #             self.cls_head.train(True)

    @abc.abstractmethod
    def _init_fm(self, cfg:mlc.ConfigDict, mode:Mode):
        pass
    
    @abc.abstractmethod
    def tokenize_sequences(self, sequences:typing.Sequence[str]):
        pass

    @abc.abstractmethod
    def sequence_representation(self, batch):
        pass

    @abc.abstractmethod
    def attention_mask(self, batch):
        pass

    @torch.no_grad()
    def predict(self, sequences:typing.Sequence[str]|typing.Mapping[str,str], device=None):
        is_mapping=False
        if isinstance(sequences, typing.Mapping):
            is_mapping = True
            keys = list(sequences.keys())
            sequences = [sequences[k] for k in keys]
        seq_batch = self.tokenize_sequences(sequences)
        if device:
            seq_batch = seq_batch.to(device)
        batch = {'seq': seq_batch}
        logits = self(batch)
        logits_norm = torch.softmax(logits, dim=-1)
        logits_norm = logits_norm[:,1].cpu()
        if is_mapping:
            return {k:logits_norm[i].item() for i,k in enumerate(keys)}
        else:
            return logits_norm

class RP3Esm2(RP3Net):
    def _init_esm2(self, cfg:mlc.ConfigDict, mode:Mode, cfg_path:str|os.PathLike) -> None:
        esm_cfg = hub.EsmConfig.from_pretrained(cfg_path, local_files_only=True)
        if (checkpoint_file := cfg.get('fm.cp')):
            log.info(f"Loading pre-trained FM from checkpoint {checkpoint_file}")
            state_dict = torch.load(util.resolve(checkpoint_file), map_location='cpu', weights_only=True)
            self.fm = hub.EsmModel.from_pretrained(
                None, 
                config=esm_cfg,
                state_dict=state_dict, 
                local_files_only=True, 
                add_pooling_layer=False
            )
        else:
            log.info(f"Loading random model")
            self.fm = hub.EsmModel(esm_cfg, add_pooling_layer=False)
        self.tokenizer = hub.EsmTokenizer.from_pretrained(cfg_path, do_lower_case=False)
        self.re_aa_x = re.compile(r"[UZOB]")
        lora_config=cfg.get('fm.lora')
        if lora_config:
            lora_config = peft.LoraConfig(**lora_config, inference_mode=(mode == Mode.Inference))
            self.fm = peft.get_peft_model(self.fm, lora_config)

    def tokenize_sequences(self, seqs:typing.Sequence[str]):
        seqs = [self.re_aa_x.sub('X', s) for s in seqs]
        return self.tokenizer(seqs, padding=True, return_tensors='pt')
    
    def sequence_representation(self, batch):
        return self.fm(**batch).last_hidden_state
    
    def attention_mask(self, batch):
        return batch['attention_mask']

    def train(self, train_mode:bool=True):
        super().train(train_mode)
        if train_mode and self.mode in Mode_Training_FM:
            self.fm.embeddings.position_embeddings.requires_grad_(False)
            self.fm.contact_head.requires_grad_(False)
        return self
    
    
class RP3Esm2_650m(RP3Esm2):
    def _init_fm(self, cfg:mlc.ConfigDict, mode:Mode):
        cfg_path = pathlib.Path(__file__).resolve().parent.parent /'fm_cfg'/'esm2_650m'
        self._init_esm2(cfg, mode, cfg_path)


RP3_DEFAULT_CONFIG = mlc.FrozenConfigDict({
    'fm': {
        'type': 'esm2_650m',
        'lora': {
            'r': 8,
            'lora_alpha': 1.0,
            'target_modules': ['query', 'key', 'value'],
            'lora_dropout': 0.1,
            'bias': 'lora_only'
        },
    },
    'aggregation': 'stp',
    'stp': {
        'seq_dim': 1280,
        'd': 256,
        'num_heads': 8,
        'layer_norm': True,
    },
    'classification_head': {
        'embedding_dim': 256,
        'bias': False,
        'end_bias': True,
        'layer_norm': False,
        'layers': {
            'd': 256,
            'n': 1
        },
        'nonlinearity': 'SiLU',
    }
})

RP3_CONFIG_B = mlc.FrozenConfigDict({
    'fm':{'type': 'esm2_650m'},
    'aggregation': 'stp',
    'stp': {
        'seq_dim': 1280,
        'd': 128,
        'num_heads': 8,
        'layer_norm': True,
    },
    'classification_head': {
        'embedding_dim': 128,
        'bias': False,
        'end_bias': True,
        'layer_norm': False,
        'layers': {
            'd': 512,
            'n': 1
        },
        'nonlinearity': 'SiLU',
    }
})


def load_model(cfg:mlc.ConfigDict, cp_path:str|os.PathLike|None=None) -> RP3Net:
    model_type = cfg.fm.type
    mode = Mode[cfg.get('mode', 'Inference')]
    if model_type == 'esm2_650m':
        model = RP3Esm2_650m(cfg)
    else:
        raise ValueError(f"Model {model_type} not supported")
    if cp_path:
        cp = torch.load(cp_path, map_location='cpu', weights_only=True)
        model.load_state_dict(cp, strict=True)
    return model
        
