import warnings
from numpy import RankWarning
from torch import save
from torchdrug.utils.loggers.base_logger import BaseLogger

try:
    import wandb
except ImportError:
    wandb = None

class WandbLogger(BaseLogger):
    def __init__(self, project=None, name=None, save_dir=None, log_interval=100, **kwargs):
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."
            )

        super().__init__(log_interval=log_interval)
        self._wandb_init = dict(
            name=name,
            project=project,
            dir=save_dir
        )
        self._wandb_init.update(**kwargs)
        self._experiment = None

        _ = self.experiment

        self.experiment.define_metric("epoch")
        self.experiment.define_metric("average/*", step_metric="epoch")
    
    @property
    def experiment(self):
        if self._experiment is None:
            if wandb.run is None:
                self._experiment = wandb.init(**self._wandb_init)
            else:
                warnings.warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
        
        return self._experiment
    
    def watch(self, model):
        self.experiment.watch(model)
    
    def log(self, record, type='train'):
        if type == 'train':
            updated_record = dict()
            for key, val in record.items():
                if key.startswith("average"):
                    key = "average/" + key.lstrip("average")
                updated_record[key] = val

            self.experiment.log(updated_record)
        else:
            self.experiment.summary.update({
                type + " " + k: v for k, v in record.items()
            })
    
    def watch(self, model):
        self.experiment.watch(model, log_graph=True)
    
    def save_hyperparams(self, params):
        self.experiment.config.update(params)
            
