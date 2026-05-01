import pandas as pd
import numpy as np
import statistics

class EarlyStopping_max:
    """ early stopping with validation dataset for regression """
    def __init__(self, tolerance=0, patience=30, valid_metric=None, train_metric=None, data_target=None, verbose=True, save=True):  #, logger=None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        if valid_metric == None:
            self.best_cv_metric = 0
        else:
            self.best_cv_metric = min(valid_metric)
        if train_metric == None:
            self.best_train_metric = 0
        else:
            self.best_train_metric = min(train_metric)
        self.ready_to_stop = False
        self.save = save
        self.data_target = data_target
    
    def update_metrics(self, cv_metrics, train_metrics):
        self.best_cv_metric = max(cv_metrics, self.best_cv_metric)
        self.best_train_metric = max(train_metrics, self.best_train_metric)
            
    def __call__(self, model, epoch, train_metrics, cv_metrics, logger, folder, final=False):
        """recommended input for train/cv_metrics = a list of values = [precision, sensitivity]"""
        ## model, epoch, train_metric, cv_metric, logger, folder, key
        # Convert from numpy array to list
        train_metrics = np.min(np.array(train_metrics))
        cv_metrics = np.min(np.array(cv_metrics))
        if epoch == 0:
            self.update_metrics(cv_metrics, train_metrics)
            self.save_checkpoint(f'cv', cv_metrics, model, epoch, folder, logger=logger)
        
        # Normal situation where at least one of the train_metric and cv_metric were increasing
        elif cv_metrics > self.best_cv_metric or train_metrics > self.best_train_metric:
            
            # If currently underfitting
            if train_metrics < cv_metrics:
                if train_metrics > self.best_train_metric:
                    self.update_metrics(cv_metrics=cv_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'train', self.best_train_metric, model, epoch, folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as train metric failed to improve while underfitting.')

            # If currently overfitting, only slightly
            elif train_metrics < cv_metrics*1.03:
                if cv_metrics > self.best_cv_metric:
                    self.update_metrics(cv_metrics=cv_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'cv', self.best_cv_metric, model, epoch, folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as model ceased to improve.')

            # If currently overfitting, severely
            else:
                self.counter += 1
                logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} due to overfitting.')

        # Loss rebounded significantly
        else:
            self.counter += 1
            logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as model ceased to improve.')
                
        if self.counter >= self.patience:
            self.ready_to_stop = True
            self.save_final(model, folder)
            pass

        if final:
            logger.record(f'\nMaximum epoch number was reached for {self.data_target}.')
            self.save_final(model, folder)
                        
    def save_final(self, model, folder):
        # model.save_component(folder, self.data_target, self.epoch)
        if self.save and hasattr(self, 'epoch'):
            model.decide_final(folder, self.data_target, self.epoch)

    def save_checkpoint(self, metric_type, metric, model, epoch, folder, logger=None):
        # 'Validation', cv_metrics, model, epoch, folder, logger=logger
        self.epoch = epoch
        # print(epoch)
        if self.save:
            # print("save_cpt really")
            if self.verbose:
                if metric_type == 'train':
                    logger.record(f'Epoch {epoch+1}: {metric_type} metric increased to {self.best_train_metric:.4f}.  Saving model ...')
                else:
                    logger.record(f'Epoch {epoch+1}: {metric_type} metric increased to {self.best_cv_metric:.4f}.  Saving model ...')
            # torch.save(model.state_dict(), f'model/{folder}/checkpoint_{self.epoch+1}.pth')
            model.save_component(folder, self.data_target, epoch)
            if epoch > 1:
                self.patience = 8
            model.remove_old_chkpt(folder, self.data_target, epoch)
        self.counter = 0
        
        
class EarlyStopping_min:
    """ early stopping with validation dataset for regression """
    def __init__(self, tolerance=0, patience=41, verbose=True, save=True,
                 valid_loss=None, train_loss=None, ):  #, logger=None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        if valid_loss == None:
            self.best_cv_metric = np.Inf
        else:
            self.best_cv_metric = valid_loss
        if train_loss == None:
            self.best_train_metric = np.Inf
        else:
            self.best_train_metric = train_loss
        self.ready_to_stop = False
        self.save = save
    
    def update_metrics(self, cv_metrics=None, train_metrics=None):
        if cv_metrics != None:
            self.best_cv_metric = min(cv_metrics, self.best_cv_metric)
        if train_metrics != None:
            self.best_train_metric = min(train_metrics, self.best_train_metric)
            
    def __call__(self, model, epoch, train_metrics, cv_metrics, logger, hp_folder, data_target, final=False):
        ## (model, epoch, train_loss, valid_loss, logger, hp_folder)
        """train_metrics & cv_metrics must be a scalar loss values"""
        if epoch == 0:
            self.update_metrics(cv_metrics, train_metrics)
            self.save_checkpoint(f'{self.data_target} Validation', cv_metrics, model, epoch, hp_folder, logger=logger)
        
        # Normal situation where both train_loss and cv_loss were reducing
        elif max(train_metrics, cv_metrics) < max(self.best_cv_metric, self.best_train_metric):
            # If currently underfitting
            if train_metrics > cv_metrics:
                if train_metrics < self.best_train_metric:
                    self.update_metrics(cv_metrics=cv_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'{self.data_target} Train', train_metrics, model, epoch, hp_folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as train loss failed to improve while underfitting.')

            # If currently overfitting, only slightly
            elif train_metrics > cv_metrics*0.98:
                if cv_metrics < self.best_cv_metric:
                    self.update_metrics(cv_metrics=cv_metrics, train_metrics=train_metrics)
                    self.save_checkpoint(f'{self.data_target} Validation', cv_metrics, model, epoch, hp_folder, logger=logger)
                else:
                    self.counter += 1
                    logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as model ceased to improve.')

            # If currently overfitting, severely
            else:
                self.counter += 1
                logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} due to overfitting.')

        # Loss rebounded significantly
        else:
            self.counter += 1
            logger.record(f'\nEarlyStopping counter for {self.data_target}: {self.counter} out of {self.patience} as model ceased to improve.')
                
        if self.counter >= self.patience:
            self.ready_to_stop = True
            self.save_final(model, hp_folder)

        if final:
            logger.record(f'\nMaximum epoch number was reached for {self.data_target}.')
            self.save_final(model, hp_folder)
            
            
    def save_final(self, model, hp_folder):
        # model.save_component(hp_folder, self.data_target, epoch)
        if self.save and hasattr(self, 'epoch'):
            model.decide_final(hp_folder, self.data_target, self.epoch)

    def save_checkpoint(self, data_target_metric_type, cv_metric, model, epoch, hp_folder, logger=None):
        # f'{self.data_target} Validation', cv_metrics, model, epoch, hp_folder, logger=logger
        self.epoch = epoch
        # print("save_cpt")
        if self.save:
            # print("save_cpt really")
            if self.verbose:
                logger.record(f'Epoch {epoch+1}: {data_target_metric_type} metric decreased to {cv_metric:.4f}.  Saving model ...')
            # torch.save(model.state_dict(), f'model/{hp_folder}/checkpoint_{self.epoch+1}.pth')
            model.save_component(hp_folder, self.data_target, epoch)
            if epoch > 1:
                self.patience = 11
        self.counter = 0