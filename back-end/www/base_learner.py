from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
import logging
import logging.handlers

"""
Base PyTorch learners
Usage:
    from base_pytorch_learner import BasePyTorchLearner

    class Learner(BasePyTorchLearner):
        def __init__(self):
            super().__init__()
            self.create_logger(log_path="../log/Learner.log")

    def fit(self, Xt, Yt, Xv=None, Yv=None):
        pass

    def predict(self, X):
        pass
"""
class BaseLearner(ABC):
    def __init__(self):
        self.model = None
        self.logger = None
        if torch.cuda.is_available:
            self.use_cuda = True
        else:
            self.use_cuda = False

    # Train the model
    # Input:
    #   Xt (pandas.DataFrame or numpy.array): predictors for training
    #   Yt (pandas.DataFrame or numpy.array): response for training
    #   Xv (pandas.DataFrame or numpy.array): predictors for validation
    #   Yv (pandas.DataFrame or numpy.array): response for validation
    # Output: None
    @abstractmethod
    def fit(self, Xt, Yt, Xv=None, Yv=None):
        pass

    # Make predictions
    # Input:
    #   X (pandas.DataFrame or numpy.array): predictors for testing
    # Output:
    #   Y (numpy.array): predicted response for testing
    @abstractmethod
    def predict(self, X):
        pass

    # Save model
    def save(self, out_path):
        if self.model is not None:
            torch.save(self.model.state_dict(), out_path)

    # Load model
    def load(self, in_path):
        if self.model is not None:
            self.model.load_state_dict(torch.load(in_path))

    # Log information
    def log(self, msg, lv="i"):
        print(msg)
        if self.logger is not None:
            if lv == "i":
                self.logger.info(msg)
            elif lv == "w":
                self.logger.warning(msg)
            elif lv == "e":
                self.logger.error(msg)
    
    # Create a logger
    def create_logger(self, log_path=None):
        if log_path is None:
            return None
        dir_name = os.path.dirname(log_path)
        if dir_name != "" and not os.path.exists(dir_name):
            os.makedirs(dir_name) # create directory if it does not exist
        handler = logging.handlers.RotatingFileHandler(log_path, mode="a", maxBytes=100000000, backupCount=200)
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.INFO)
        for hdlr in logger.handlers[:]:
            logger.removeHandler(hdlr) # remove old handlers
        logger.addHandler(handler)
        self.logger = logger
