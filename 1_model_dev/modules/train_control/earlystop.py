import pandas as pd
import numpy as np
import statistics
from modules.train_log import LogBook
from modules.train_control.parameters import Parameters
from modules.train_control.hp_tuning import HyperparameterTuner_Instruction

class EarlyStopping_Report:
    def __init__(self, best_epoch, last_epoch, least_overfit, earlystop_counter, ready_to_stop, patient_index, best_train_metric, best_valid_metric):
        self.best_epoch = best_epoch
        self.last_epoch = last_epoch
        self.least_overfit = least_overfit
        self.earlystop_counter = earlystop_counter
        self.ready_to_stop = ready_to_stop
        self.patient_index = patient_index
        self.best_train_metric = best_train_metric
        self.best_valid_metric = best_valid_metric

class EarlyStopping:
    """ early stopping with validation dataset for regression """
    def __init__(self, params: Parameters|None=None, 
                 train_metric=None, valid_metric=None, 
                 verbose=True, save=True, status_instructions:pd.Series|None=None, logbook:LogBook|None=None):
        self.mode = params.EARLYSTOP_MODE
        # Direction multiplier: 1 for max (higher is better), -1 for min (lower is better)
        self.d = 1 if self.mode == 'max' else -1
        self.least_overfit = status_instructions["least_overfit"] if status_instructions is not None else np.Inf
        # Threshold
        self.SLIGHT_OVERFIT_THRES = params.SLIGHT_OVERFIT_THRES
        # Mode-specific settings
        self._npmin_if_max = np.min if self.mode == 'max' else np.max
        self._max_if_max = max if self.mode == 'max' else min
        self._log_word = 'increased' if self.mode == 'max' else 'decreased'
        
        self.verbose = verbose
        self.counter = status_instructions["earlystop_counter"] if status_instructions is not None else 0
        self.best_epoch = status_instructions["best_epoch"] if status_instructions is not None else 0
        self.patience = (params.INIT_PATIENCE, params.PATIENCE)
        self.patience_index = 1 if self.best_epoch > 0 else 0
        if status_instructions["cycle_num"] > 0 and logbook!=None:
            self.read_best_metrics_from_logbook(params, logbook)
        else:
            if valid_metric == None:
                self.best_valid_metric = 0 if self.mode == 'max' else np.Inf
            else:
                self.best_valid_metric = self._max_if_max(valid_metric)
            if train_metric == None:
                self.best_train_metric = 0 if self.mode == 'max' else np.Inf
            else:
                self.best_train_metric = self._max_if_max(train_metric)
        self.ready_to_stop = False
        self.save = save
        self.eval_metrics = params.EARLYSTOP_METRICS

    def read_best_metrics_from_logbook(self, params:Parameters, logbook:LogBook):
        n = len(params.EARLYSTOP_METRICS)
        self.best_train_metric = self._max_if_max(list(self._npmin_if_max(logbook.df.loc[(logbook.df["saved_chpt"]==True) & (logbook.df["train_valid_test"]=="train"), 
                                                                       params.EARLYSTOP_METRICS], 
                                                                       axis=1)))
        self.best_valid_metric = self._max_if_max(list(self._npmin_if_max(logbook.df.loc[(logbook.df["saved_chpt"]==True) & (logbook.df["train_valid_test"]=="valid"), 
                                                                       params.EARLYSTOP_METRICS], 
                                                                       axis=1)))
        # To obtain least_overfit, we need to calculate self.d * (train-valid)
        train_metrics_np = self._npmin_if_max(logbook.extract_metric_1phase(params=params, phase="train"), axis=1)
        valid_metrics_np = self._npmin_if_max(logbook.extract_metric_1phase(params=params, phase="valid"), axis=1)
        all_overfit_record = self.d * (train_metrics_np - train_metrics_np)
        self.least_overfit = np.min(all_overfit_record).item()

    def update_metrics(self, valid_metrics, train_metrics):
        self.best_train_metric = self._max_if_max(train_metrics, self.best_train_metric)
        self.best_valid_metric = self._max_if_max(valid_metrics, self.best_valid_metric)
            
    def __call__(self, model, epoch, train_metrics, valid_metrics, logger, folder, final=False, hp_tuning:bool=False):
        """ Recommended input for train/valid_metrics = a list of values = [precision, sensitivity] """
        ## model, epoch, train_metric, valid_metric, logger, folder, key
        # Convert from numpy array to list
        train_metrics = self._npmin_if_max(np.array(train_metrics))
        valid_metrics = self._npmin_if_max(np.array(valid_metrics))
        self.saved = False # Did we save this time?
        
        current_overfit = self.d * (train_metrics - valid_metrics)   # self.d = 1 if "max" else -1
        
        if epoch == 0:
            self.update_metrics(valid_metrics, train_metrics)
            self.save_checkpoint(f'valid', valid_metrics, model, epoch, folder, logger=logger)

        # Normal situation where at least one of the train_metric and valid_metric were improving
        elif self.d * (valid_metrics - self.best_valid_metric) > 0 or \
             self.d * (train_metrics - self.best_train_metric) > 0:
            
            # If currently underfitting
            if current_overfit < 0:
                if self.d * (train_metrics - self.best_train_metric) > 0:
                    self.update_metrics(valid_metrics=valid_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'train', self.best_train_metric, model, epoch, folder, logger=logger)
                    
                # In some cases, train_metric marginally didn't improve, but valid_metric improved a lot
                elif self.d * (train_metrics - self.best_train_metric) * 10 <  self.d * (valid_metrics - self.best_valid_metric):
                    self.update_metrics(valid_metrics=valid_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'valid', self.best_valid_metric, model, epoch, folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter: {self.counter} out of {self.patience[self.patience_index]} as train metric failed to improve while underfitting.') # for {self.data_target}

            # If currently overfitting, only slightly (dp on whether the last epoch train or valid higher)
            elif self._is_slight_overfit(train_metrics, valid_metrics):
                done = False
                # If (best_valid is better than best_train metric) and (overall improving) in last epoch
                if self.d * (self.best_valid_metric - self.best_train_metric) > 0:
                    if self.d * ((valid_metrics + train_metrics) - (self.best_valid_metric + self.best_train_metric)) > 0:
                        self.update_metrics(valid_metrics=valid_metrics, train_metrics=train_metrics)
                        self.save_checkpoint(f'train', self.best_train_metric, model, epoch, folder, logger=logger)
                        done= True
                # If valid_metric improved
                if not done and self.d * (valid_metrics - self.best_valid_metric) > 0:
                    self.update_metrics(valid_metrics=valid_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'valid', self.best_valid_metric, model, epoch, folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter: {self.counter} out of {self.patience[self.patience_index]} as model ceased to improve.')

            # If currently overfitting severely, but overfit reducing AND valid_metric improving (usualy happen in the first 3 epoch)
            elif self.d * (valid_metrics - self.best_valid_metric) > 0 and current_overfit < self.least_overfit:
                self.update_metrics(valid_metrics=valid_metrics, train_metrics=train_metrics)
                self.save_checkpoint(f'valid', self.best_valid_metric, model, epoch, folder, logger=logger)

            # If currently overfitting, no notable improvement
            else:
                self.counter += 1
                logger.record(f'\nEarlyStopping counter: {self.counter} out of {self.patience[self.patience_index]} due to overfitting.')

        # Loss rebounded significantly AND neither train or valid metrics improved
        else:
            self.counter += 1
            logger.record(f'\nEarlyStopping counter: {self.counter} out of {self.patience[self.patience_index]} as model ceased to improve.')
            #  for {self.data_target}

        self.least_overfit = min(self.least_overfit, current_overfit)   
        logger.record(f"current_overfit = {current_overfit} --> self.least_overfit = {self.least_overfit}")
        
        if self.counter >= self.patience[self.patience_index]:
            logger.record(f"Max Earlystopping tolerance reached: stop now")
            self.ready_to_stop = True
            self.decide_final(model, folder, self.best_epoch, hp_tuning)
            pass

        if final:
            logger.record(f'\nMaximum epoch number was reached.') #  for {self.data_target}
            if not self.saved:
                model.save_chkpt(folder, epoch, logger=logger)
                
    
    def _is_slight_overfit(self, train_metrics, valid_metrics):
        """Check if model is only slightly overfitting (mode-specific logic)"""
        if self.mode == 'max':
            # Max mode: metrics bounded [0, 1], use both multiplicative and additive thresholds
            return train_metrics < valid_metrics * (1 + self.SLIGHT_OVERFIT_THRES) or \
                   (train_metrics < valid_metrics + self.SLIGHT_OVERFIT_THRES * 2 and valid_metrics < 0.5)
        else:
            # Min mode: loss unbounded, use only multiplicative threshold
            return train_metrics < valid_metrics * (1 - self.SLIGHT_OVERFIT_THRES)
                        
    def decide_final(self, model, folder, epoch, hp_tuning:bool):
        if self.save:
            model.decide_final(hp_folder=folder, epoch=epoch, hp_tuning=hp_tuning)

    def save_checkpoint(self, metric_type, metric, model, epoch, folder, logger=None):
        """ This is only trigger only when better checkpoint was found. To save due to final epoch, just call model.save_chkpt(foler, epoch) directly """
        self.best_epoch = epoch
        self.saved = True
        if self.save:
            if self.verbose:
                if metric_type == 'train':
                    logger.record(f'Epoch {epoch+1}: {metric_type} metric {self._log_word} to {self.best_train_metric:.4f}.  Saving model ...')
                else:
                    logger.record(f'Epoch {epoch+1}: {metric_type} metric {self._log_word} to {self.best_valid_metric:.4f}.  Saving model ...')
            model.save_chkpt(folder, epoch, logger=logger)
            if epoch > 1:
                self.patience_index = 1
            model.remove_old_chkpt(folder, epoch)
        self.counter = 0

    def export_report(self, epoch):
        # print("When export EarlyStopping_Report, least_overfit =", self.least_overfit)
        return EarlyStopping_Report(
            best_epoch=self.best_epoch, 
            last_epoch=epoch, 
            least_overfit=self.least_overfit, 
            earlystop_counter=self.counter, 
            ready_to_stop=int(self.ready_to_stop), 
            patient_index=self.patience_index, 
            best_valid_metric = self.best_valid_metric, 
            best_train_metric = self.best_train_metric
        )
        