import importlib
from RP3Net.training.cli import RP3Cli
from RP3Net.util import util

log = util.get_logger(__name__)


def rp3_train():
    try:
        assert importlib.util.find_spec('lightning') is not None, "Please install 'RP3Net[training]' to enable training"
        cli = RP3Cli()
    except Exception as e:
        log.error("Top level catch", exc_info=e)
        raise e
    

if __name__ == "__main__":
    rp3_train()
