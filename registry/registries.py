import importlib
import logging
import os
import sys
from copy import deepcopy

from .registry import Registry
from .configs import MODULES


class Registries():  # pylint: disable=invalid-name, too-few-public-methods
    """All module Registries."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")
    
    score = Registry("flow_score") # score the importance of a frame
    strategy = Registry("strategy") # the strategy used to select frame
    evaluation = Registry("evaluation") # evaluating method for each strategy
    model = Registry("model") # models for evaluating

    @classmethod
    def import_all_modules(cls):
        current_work_dir = os.getcwd()
        if current_work_dir not in sys.path:
            sys.path.append(current_work_dir)
        """Import all modules for registry."""
        all_modules = deepcopy(MODULES)
        for base_dir, modules in all_modules:
            for name in modules:
                try:
                    if base_dir != "":
                        full_name = base_dir + "." + name
                    else:
                        full_name = name
                    importlib.import_module(full_name)
                    logging.debug(f"{full_name} loaded.")
                except ImportError as error:
                    logging.warning((name, error))
