import re
import os
import socket
import sys
import logging
import lightning.pytorch as L
import lightning.pytorch.utilities as L_util
import lightning.pytorch.cli as L_cli
import lightning.pytorch.callbacks as L_cb
import lightning.pytorch.loggers as L_log
import wandb

from . import lm
from .. import util

log = util.get_logger(__name__)

def setup_logging_torch(args):
    log_level = args.log_level
    os.environ["PP_LOG_LEVEL"] = log_level
    logfile = args.logfile
    if logfile is not None:
        logfile_base = util.resolve(logfile)
        logfile_base = str(logfile_base.parent/logfile_base.stem)
        os.environ["PP_LOGFILE_BASE"] = logfile_base
        # /homes/evgeny/micromamba/envs/ai/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py
        # SLURMEnvironment.world_size() and SLURMEnvironment.global_rank()
        if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
            logfile = logfile.replace(".log", f"_{os.environ.get('SLURM_PROCID', '0')}.log")
    util.setup_logging(logfile, log_level, log_console=logfile is None)
    ll = logging.getLogger("lightning")
    ll.propagate = True
    ll.handlers.clear()
    ll = logging.getLogger("lightning.pytorch")
    ll.handlers.clear()
    ll.propagate = True
    log.info(f"Host: {socket.gethostname()}; PID: {os.getpid()}; Command line: {' '.join(sys.argv)}")

class RP3Cli(L_cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        self.wandb_logger = None
        super().__init__(*args, **{**kwargs, 'save_config_kwargs': {"overwrite": True}})

    @staticmethod
    @L_util.rank_zero_only
    def wandb_init(wandb_project, wandb_run_name, wandb_run_id):
        return wandb.init(id=wandb_run_id, project=wandb_project, name=wandb_run_name, resume='allow')

    def wandb_logger_init(self, config):
        if 'wandb' not in config or config.wandb is None or \
            'project' not in config.wandb or ('run' not in config.wandb and 'run_id' not in config.wandb) or \
                config.wandb.project is None or (config.wandb.run is None and config.wandb.run_id is None) or \
                ('disable' in config.wandb and config.wandb.disable):
            log.info("No wandb logging")
            return
        else:
            log.info("Configure wandb logging")
        run_id = config.wandb.run_id if 'run_id' in config.wandb and config.wandb.run_id is not None \
            else config.wandb.run
        run = self.wandb_init(wandb_project=config.wandb.project, wandb_run_name=config.wandb.run, wandb_run_id=run_id)
        if run is not None:
            log.info(f"Wandb run: {run.name}({run.id})")
        logger = L_log.WandbLogger(project=config.wandb.project, name=config.wandb.run, id=run_id)
        return logger

    def add_arguments_to_parser(self, parser: L_cli.LightningArgumentParser) -> None:
        parser.add_argument("--logfile", help="Log file. Log output to console if set to None.")
        parser.add_argument("--log_level", default="info",
                            help="Log level of root logger. Appender levels are appropriately hard coded.")
        parser.add_argument("--track_metric_checkpoints", choices=["last", "all", "best"],
                            help="""
Track checkpoints for training and validation metrics from the module. 
If not provided, no checkpoints will be recorded at all.
If any value is provided, only the best checkpoint will be recorded for all the metrics specified by `model.metrics_for_checkpointing()`.
The value of this argument affects how checkpoints will be saved for 'train_loss'.
                            """)
        parser.add_argument("--wandb.project", help="Wandb project name", default=None)
        parser.add_argument("--wandb.run", help="Wandb run name", default=None)
        parser.add_argument("--wandb.run_id", help="Wandb run id, same as name by default", default=None)
        parser.add_argument("--wandb.disable", help="Set to true to turn off wandb logging, without removing the rest of wandb settings", action='store_true')
        parser.add_argument("--test_after_fit_metric", help="Metric to use for test_after_fit. If not set, then do not run test_after_fit", default=None)
        parser.add_argument("--emlc_k", help="Number of student gradinent steps to perform per teacher step for EMLC", default=1, type=int)

    def before_instantiate_classes(self) -> None:
        config = self.config.get(str(self.subcommand), self.config)
        setup_logging_torch(config)
        self.wandb_logger = self.wandb_logger_init(config)

    
    def instantiate_trainer(self, **kwargs) -> L.Trainer:
        log.info("Instantiating trainer")
        config = self.config.get(str(self.subcommand), self.config)
        metric_checkpoints = self._get(self.config_init, 'track_metric_checkpoints')
        # metric_checkpoints = bool(metric_checkpoints)
        if metric_checkpoints is not None:
            self.init_metric_checkpoints(metric_checkpoints)
        self.add_loggers(config)
        return super().instantiate_trainer(**kwargs)

    def add_loggers(self, config):
        configured_loggers = self._get(self.config_init, 'trainer.logger', default=[])
        if configured_loggers == True or configured_loggers is None:
            configured_loggers = []
        elif isinstance(configured_loggers, L_log.Logger):
            configured_loggers = [configured_loggers]
        elif configured_loggers == False:
            return
        add_csv_logger = True
        for logger in configured_loggers:
            if isinstance(logger, L_log.CSVLogger):
                add_csv_logger = False
                break
        if add_csv_logger:
            csv_logger = L_log.CSVLogger(config['trainer']['default_root_dir'])
            configured_loggers.append(csv_logger)
        if self.wandb_logger is not None:
            configured_loggers.append(self.wandb_logger)
        config_init = self.config_init.get(str(self.subcommand), self.config_init)
        config_init['trainer']['logger'] = configured_loggers

    def init_metric_checkpoints(self, checkpoint_save_flag):
        model: lm.RP3LM = self.model
        metrics = model.metrics.metrics_for_checkpointing()
        default_root_dir = util.resolve(self._get(self.config, 'trainer.default_root_dir'))
        if metrics is None:
            return
        trainer_config = self._get(self.config_init, 'trainer', default={})
        if 'callbacks' not in trainer_config or trainer_config['callbacks'] is None:
            callbacks = []
            trainer_config['callbacks'] = callbacks
        else:
            callbacks = trainer_config['callbacks']
            if isinstance(callbacks, L.Callback):
                callbacks = [callbacks]
                trainer_config['callbacks'] = callbacks
        for i, (key, metric) in enumerate(metrics.items()):
            mode = 'max' if metric.higher_is_better else 'min'
            callbacks.append(L_cb.ModelCheckpoint(
                dirpath=default_root_dir, monitor=key, mode=mode,
                filename='{epoch}_{'+key+':.2f}',
                save_on_train_epoch_end=False, save_weights_only=False
            ))
        assert checkpoint_save_flag in ['last', 'all', 'best']
        if checkpoint_save_flag == 'best':
            save_top_k = 1
        elif checkpoint_save_flag == 'all':
            save_top_k = -1
        else:
            save_top_k = 0
        callbacks.append(L_cb.ModelCheckpoint(dirpath=default_root_dir, monitor='train_loss', filename='{epoch}_{train_loss:.2f}', mode='min',
                                                      save_top_k=save_top_k, save_last=True))

    def test_after_fit(self, metric):
        model: lm.RP3LM = self.model
        dm = self.datamodule
        dir = util.resolve(self.trainer.default_root_dir)
        filename_pattern = re.compile(r'^epoch=\d+_' + metric + r'=\d+(\.\d+)?\.ckpt$')
        cp_file = util.find_checkpoint_file(dir, filename_pattern)
        log.info(f"Loading checkpoint {cp_file}")
        self.trainer.test(model, dm, ckpt_path=str(cp_file))

    def after_fit(self):
        metric = self._get(self.config, 'test_after_fit_metric')
        if metric is not None:
            log.info(f"Running test_after_fit on {metric}")
            self.test_after_fit(metric)
