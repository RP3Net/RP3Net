from .model import RP3Net, RP3_DEFAULT_CONFIG, RP3_CONFIG_B, load_model
from .rp3_main import rp3_main
import importlib
if importlib.util.find_spec('lightning') is not None:
    from .rp3_train import rp3_train
else:
    def rp3_train():
        raise ImportError("Please install 'RP3Net[training]' to enable training")
