import typing
import itertools
import torch
import torch.distributed as dist

def _torch_split_mask(o, mask:torch.Tensor):
    if isinstance(o, typing.Mapping):
        return {k:_torch_split_mask(v, mask) for k,v in o.items()}
    elif isinstance(o, torch.Tensor):
        return o[mask]
    elif isinstance(o, typing.Sequence):
        return list(itertools.compress(o, mask.tolist()))
    else:
        raise ValueError(f"Unsupported type {type(o)}")

def torch_split_key_index(batch:typing.Mapping, key:str, ix_first:torch.Tensor):
    assert key in batch
    assert isinstance(batch[key], torch.Tensor)
    assert batch[key].dim() == 1 and ix_first.dim() == 1
    assert batch[key].dtype == ix_first.dtype
    ix_tz = (batch[key].unsqueeze(1) == ix_first.unsqueeze(0)).any(dim=1)
    batch1 = _torch_split_mask(batch, ix_tz)
    batch2 = _torch_split_mask(batch, ~ix_tz)
    return batch1, batch2

def torch_split_size(batch:typing.Mapping, size:tuple[int]):
    ix_tz = torch.zeros(sum(size), dtype=torch.bool)
    start = 0
    ret = []
    for s in size:
        ix = ix_tz.clone()
        ix[start:start+s] = True
        ret.append(_torch_split_mask(batch, ix))
        start += s
    return tuple(ret)

def _find_size(o):
    if isinstance(o, typing.Mapping):
        k = next(iter(o.keys()))
        return _find_size(o[k])
    elif isinstance(o, torch.Tensor):
        return o.shape[0]
    elif isinstance(o, typing.Sequence):
        return len(o)
    else:
        raise ValueError(f"Unsupported type {type(o)}")
    
def torch_split_chunks(batch:typing.Mapping, n_chunks:int, total_size:int=-1):
    if total_size == -1:
        total_size = _find_size(batch)
    chunk_size = total_size // n_chunks
    remdr = total_size % n_chunks
    ix_tz = torch.zeros(total_size, dtype=torch.bool)
    start = 0
    ret = []
    while start < total_size:
        ix = ix_tz.clone()
        cs = chunk_size
        if remdr:
            cs += 1 
            remdr -= 1
        end = min(start + cs, total_size)
        ix[start:end] = True
        ret.append(_torch_split_mask(batch, ix))
        start = end
    return tuple(ret)

def _torch_concat_chunks(sentinel, chunks):
    if isinstance(sentinel, typing.Mapping):
        return {k:_torch_concat_chunks(sentinel[k], [c[k] for c in chunks]) for k in sentinel.keys()}
    elif isinstance(sentinel, torch.Tensor):
        return torch.cat(chunks, dim=0)
    elif isinstance(sentinel, typing.Sequence):
        return list(itertools.chain(*chunks))
    else:
        raise ValueError(f"Unsupported type {type(sentinel)}")

def torch_concat_chunks(chunks:typing.Iterable[typing.Mapping]):
    if len(chunks) == 0:
        return {}
    sentinel = next(iter(chunks))
    return _torch_concat_chunks(sentinel, chunks)

def is_distr_env():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True