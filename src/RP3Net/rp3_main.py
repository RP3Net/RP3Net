
from typing import Optional
import yaml
import argparse
import logging
from tqdm import tqdm
import ml_collections as mlc
import pandas as pd
import numpy as np

import RP3Net.util as util
import RP3Net.model as model

log = util.get_logger(__file__)


def setup_args(_args: Optional[list] = None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""
    Predict protein expression in E.coli from sequences. Takes in a fasta file with protein sequences and ids.
    Outputs a .csv[.gz] file with "id,score" columns. 
    The score is the predicted probability of expression.
    """)
    parser.add_argument("--log_file", help="Log file. Log output to console if set to None.")
    parser.add_argument("--log_level", default="info", help="Log level of root logger. Appender levels are appropriately hard coded.")
    parser.add_argument("-c", "--config", help="Model configuration in yaml, or registry entry.", default="RP3_DEFAULT_CONFIG")
    parser.add_argument("-p", "--checkpoint", required=True, help="Model checkpoint.")
    parser.add_argument("-f", "--fasta", required=True, help="Fasta file, possibly gzipped.")
    parser.add_argument("-d", "--device", default='cpu', help="Device to use: cpu, cuda[:n], mps, ...")
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="Batch size. Memory consumption depends on construct sequence length and batch size. If the model runs out of SLURM/GPU memory, try reducing this parameter to make it fit.")
    parser.add_argument("-o", "--out_file", required=True, help="Output .csv[.gz] file.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show progress bar.")
    return parser.parse_args(_args)

MODEL_REGISTRY = {
    'RP3_DEFAULT_CONFIG': model.RP3_DEFAULT_CONFIG,
    'RP3_CONFIG_B': model.RP3_CONFIG_B,
}

def rp3_main():
    args = setup_args()
    util.setup_logging(save_path=args.log_file, level=args.log_level, log_console=args.log_file is None)

    seq_map = util.read_fasta(args.fasta)
    seq_lens = np.array(list(map(len, seq_map.values())))
    log.info(f"Read {len(seq_map)} sequences from {args.fasta}. Sequence lengths (mean/std/median/min/max): "
             f"{seq_lens.mean():.2f}/{seq_lens.std():.2f}/{np.median(seq_lens):.0f}/{seq_lens.min()}/{seq_lens.max()}")
    
    config_path = util.resolve(args.config)
    if config_path.exists():
        with open(config_path, "r") as f: 
            config = mlc.FrozenConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    elif args.config in MODEL_REGISTRY:
        config = MODEL_REGISTRY[args.config]
    else:
        raise ValueError(f"Config file {args.config} not found.")
    log.info(f"Loading model {args.config} from checkpoint {args.checkpoint}")
    m = model.load_model(config, args.checkpoint)
    m = m.to(device=args.device)
    m = m.eval()
    log.info(f"Loaded model {m}")
    
    seq_keys = list(seq_map.keys())
    def batches():
        if args.progress:
            tqdm_desc = args.fasta.replace('.fasta', '').replace('.gz', '')[-20:]
            r = tqdm(range(0, len(seq_keys), args.batch_size), desc=tqdm_desc)
        else:
            r = range(0, len(seq_keys), args.batch_size)
        for i in r:
            if not args.progress:
                log.info(f"Processing batch {i // args.batch_size + 1} of {len(seq_keys) // args.batch_size + 1}.")
            yield {k: seq_map[k] for k in seq_keys[i:i + args.batch_size]}

    ret = {}
    for b in batches():
        ret.update(m.predict(b, device=args.device))
    keys, scores = zip(*ret.items())
    out_df = pd.DataFrame({'id': keys, 'score': scores})
    log.info(f"Writing {out_df.shape[0]} rows to {args.out_file}")
    out_df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    rp3_main()

