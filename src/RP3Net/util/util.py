import logging
import typing
import sys
import pathlib
import re
import os


def get_logger(name: str) -> logging.Logger:
    log_key = name[name.rfind('.')+1:] if '.' in name else name
    return logging.getLogger(log_key)

log = get_logger(__file__)

def resolve(filename) -> pathlib.Path:
    if type(filename) == str:
        filename = pathlib.Path(filename)
    if str(filename).startswith('~'):
        filename = filename.expanduser()
    return filename.resolve()


def find_checkpoint_file(dir:os.PathLike, filename_pattern:re.Pattern) -> os.PathLike:
    cp_files = list(dir.glob(f"*.ckpt"))
    cp_files = [f for f in cp_files if filename_pattern.match(f.name)]
    if len(cp_files) == 0:
        raise RuntimeError(f"No checkpoint files for metric '{filename_pattern}' in {dir}")
    if len(cp_files) > 1:
        log.warning(f"Multiple checkpoint files for metric '{filename_pattern}' in {dir}. Will use {cp_files[0]}")
    cp_file = cp_files[0]
    return cp_file

def setup_logging(save_path = None,
                  level: typing.Union[str, int] = 'info',
                  log_console: bool = True,
                  formatter: typing.Optional[logging.Formatter] = None) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    else:
        level = level

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if formatter is None:
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s.%(msecs)03d [%(process)d:%(threadName)s] <%(name)s> - %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S")

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    if log_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if save_path is not None:
        file_handler = logging.FileHandler(resolve(save_path))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

