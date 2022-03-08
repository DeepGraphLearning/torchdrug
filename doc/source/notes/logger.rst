Custom Loggers
==============

.. include :: ../bibliography.rst

TorchDrug also supports using 3rd party and custom loggers to log metrics during training.

All the loggers are derived from the the :class:`core.LoggerBase <torchdrug.core.LoggerBase>` class.

Define your own logger
----------------------

.. code-block:: python
    
    from torchdrug.core import LoggerBase

    class CustomLogger(LoggerBase):
        def __init__(self):
            # Define the logger constructor
            pass
        
        def log(self, record, step_id, category="train/batch"):
            """
            record is a dict of the metrics
            step_id is the index of the log step
            category is the category to which the metrics belong.
                 Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``
            """
            pass
        
        def log_config(self, config):
            """
            This logs the hyperparameters of the ongoing experiment
            """
            pass

To define a custom logger, the user must define these three functions.
Using the logger requires passing it as an argument to the engine.

.. code-block:: python

    from torchdrug import core
    custom_logger = CustomLogger()

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         batch_size=1024, gpus=[0], logger=custom_logger)

Console Logging
---------------

The default logger is the console logger which logs all the information
to the console. It is always enabled along with any other custom logger.

Logging metrics to Weights & Biases
-----------------------------------

To use the `W&B`_ logger to log training metrics, install the wandb client
and login to your wandb account using the following commands.

.. code:: bash

    pip install wandb
    wandb.login()

To use wandb for logging metrics along with the console logger

.. code:: python

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         batch_size=1024, gpus=[0], logger='wandb')

For further customization of the wandb run

.. code:: python
    
    from torchdrug.core import WandbLogger
    wandb_logger = WandbLogger(project="<project_name>", name="<run_name>")

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         batch_size=1024, gpus=[0], logger=wandb_logger)


An example of the application of W&B to TorchDrug can be found `here`_ .

.. _here: https://wandb.ai/manan-goel/GCPN/reports/De-Novo-Molecule-Generation-with-GCPNs-using-TorchDrug--VmlldzoxNDgzMzQz
