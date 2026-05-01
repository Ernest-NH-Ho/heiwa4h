import numpy as np
from scipy.stats import spearmanr, pearsonr

class Metric_tuner_double_bounded:
    """ Suitable as dropout tuner, as dropout is double-bounded between 1 and 0, or other similar metrics """
    
    def __init__(self, metric, metric_upper_limit, metric_lower_limit, thres_Q2, thres_Q1, max_step, epoch_scale, eval_interval, mode):
        """ step = how many epoch will the model wait update dropout rate, do not set step less than 4 """
        self.metric = metric
        self.last_direction = None
        self.last_overfit_average = None
        self.adjusting = False
        self.last_mode = 'fine'
        self.coarse_chase = 0
        self.eval_interval = eval_interval
        self.thres_Q2 = thres_Q2
        self.thres_Q1 = thres_Q1
        self.max_step = max_step
        self.metric_upper_limit = metric_upper_limit
        self.metric_lower_limit = metric_lower_limit
        self.epoch_scale = epoch_scale
        self.mode = mode
        self.initialize_step()
        
    def initialize_step(self):
        """Assume fine mode"""
        self.step_up = max(0.00001, min(self.metric, self.metric_upper_limit-self.metric)/8)
        self.step_dn = self.step_up
        
    def set_coarse_steps(self):
        """When coarse adjustment mode were necessary"""
        self.step_up = max(0, min(self.metric, self.metric_upper_limit-self.metric)/4)
        self.step_dn = self.step_up
    
    def adjust_step_coarse(self):
        """If coarse mode, this is called. 
        If previous mode is coarse, self.adjusting=True -> adjust step according the outcome of the last tunning.
        Otherwise, fine-adjust the steps"""
        if self.last_mode == 'coarse':  # self.adjusting must be in this case
            
            # If reversed
            if self.direction == -self.last_direction:
                if self.last_direction<0:
                    self.step_up = self.step_dn/2
                else:
                    self.step_dn = self.step_up/2
                self.coarse_chase = 0
                
            # If still worsening or follow the same trend
            else:
                self.coarse_chase += 1
                if self.metric < self.metric_lower_limit + (self.metric_upper_limit - self.metric_lower_limit) * 0.05:
                    self.step_up *= max((np.average(self.overfit)/self.last_overfit_average)**3, 1+self.max_step)
                    self.step_dn = min(self.metric, 1-self.metric)/4  # *(np.average(self.overfit)/self.last_overfit_average)**2
                elif self.metric > self.metric_upper_limit * 0.9:
                    self.step_up = min(self.metric, 1-self.metric)/4  # *(np.average(self.overfit)/self.last_overfit_average)**2
                    self.step_dn *= max((np.average(self.overfit)/self.last_overfit_average)**3, 1+self.max_step)
                elif self.coarse_chase >= 1:
                    self.step_up = min(self.step_up*(np.average(self.overfit)/self.last_overfit_average)**2, min(self.metric, 1-self.metric)/4)
                    self.step_dn = self.step_dn
        else:
            self.set_coarse_steps()
            self.coarse_chase = 0
        self.adjusting = True
    
    def adjust_step_fine(self):
        """If fine mode, this is called. 
        If previous mode is coarse, reset steps. 
        Otherwise, fine-adjust the steps"""
        if self.adjusting:
            if self.last_mode == 'coarse':
                self.initialize_step()
            else:
                # If reversed
                if self.direction == -self.last_direction:
                    if self.last_direction<0:
                        self.step_up = self.step_dn/2
                    else:
                        self.step_dn = self.step_up/2
                # If still worsening
                else:
                    self.step_up *= (np.average(self.overfit)/self.last_overfit_average)**2
                    self.step_dn *= (np.average(self.overfit)/self.last_overfit_average)**2
                    
    def address_overfit_fine(self, logger):
        logger.record(f'Address overfitting using fine mode')
        self.adjust_step_fine()
        if self.metric_upper_limit != np.Inf:
            if self.metric<self.metric_upper_limit*0.95:
                self.metric = min(self.metric_upper_limit*0.98, self.metric + self.direction*self.step_up)
            else:
                self.metric = min(self.metric_upper_limit*0.999, self.metric + self.direction*self.step_up)
        else:
             self.metric = self.metric + self.direction*self.step_up
        self.adjusting = True
        self.last_mode = 'fine'
        
    def address_underfit_fine(self, logger):
        logger.record(f'Address underfitting using fine mode')
        self.adjust_step_fine()
        self.metric = max(0, self.metric + self.direction*self.step_dn)
        self.adjusting = True
        self.last_mode = 'fine'
    
    def identify_trend(self, r, rp):
        """This trend number will decide whether dynamic mode will be used for updating metric"""
        if r >= 0.6 and rp >= 0.4:
            self.trend = 1
        elif r <= -0.6 and rp <= -0.4:
            self.trend = -1
        else:
            self.trend = 0
    
    def calc_slope(self):
        """Invoked during dynamic mode"""
        self.overfit_slope = (np.average(self.overfit[-2:]) - np.average(self.overfit[:2]))/(self.eval_interval-2)
    
    def updateMetric_xTrend(self, logger):
        # Overfit (One of the conditions: 1. cv above train, 2. trend was clear even if cv was similar to train)
        if (np.percentile(self.overfit, 25) > self.thres_Q1 or \
            np.percentile(self.overfit, 50) > self.thres_Q2) and \
        self.r.correlation >= -0.6:
            # Coarse mode necessary
            if np.average(self.overfit)>self.thres_Q2:
                logger.record(f'Address overfitting using coarse mode')
                self.adjust_step_coarse()
                logger.record(f'Consecutive coarse adjustment: {self.coarse_chase}')
                if self.metric_upper_limit != np.Inf:
                    self.metric = min(self.metric_upper_limit*0.98, self.metric + self.direction*self.step_up)
                else:
                    self.metric = self.metric + self.direction*self.step_up
                self.last_mode = 'coarse'
            # Fine mode
            else:
                self.address_overfit_fine(logger)
                
        else:
            self.initialize_step()
            self.adjusting = False
            self.coarse_chase = 0
    
    def updateMetric_Trend(self, logger):
        """avg_valid_losses is numpy array"""
        # Overfiting
        if self.trend > 0: # = 2 and np.max(self.overfit)==self.overfit[-1] and np.max(self.overfit)>-np.average(avg_valid_losses)*0.02:
            expected_epoch = (self.thres_Q2 - np.average(self.overfit))/self.overfit_slope
            # print(expected_epoch)
            # print(self.max_step - expected_epoch * self.max_step/10)
            # print(self.metric * (self.max_step - expected_epoch * self.max_step/10))
            if expected_epoch >= self.epoch_scale:
                self.step_up = max(0.000025, min(self.metric, self.metric_upper_limit-self.metric) * min(0, self.max_step - expected_epoch * self.max_step/self.epoch_scale))
            else:
                self.step_up = max(0.000025, min(self.metric, self.metric_upper_limit-self.metric) * max(0, self.max_step - expected_epoch * self.max_step/self.epoch_scale))
                
            logger.record(f'Address overfitting using dynamic mode')
            logger.record(f'Consecutive coarse adjustment: {self.coarse_chase}')
            if self.metric_upper_limit != np.Inf:
                self.metric = min(self.metric_upper_limit*0.98, self.metric + self.step_up)
            else:
                self.metric = self.metric + self.step_up
            self.last_mode = 'coarse'
                
    def __call__(self, train_metric:list, valid_metric:list, logger):
        """mode = 'max' or 'min'"""
        """max mode use f1 score as metric; min mode use loss as metric"""
        train_metric = np.array(train_metric)
        valid_metric = np.array(valid_metric)
        # avg = min(np.mean(train_metric), np.mean(valid_metric))
        # train_metric /= avg
        # valid_metric /= avg
        if self.mode == 'max':
            self.overfit = train_metric / (valid_metric + 1e-12)
        elif self.mode == 'min':
            self.overfit = valid_metric / (train_metric + 1e-12) 
        else:
            print(f"ERROR: mode must be either 'max' or 'min'.")
            quit()
        self.r = spearmanr(self.overfit, b=np.array(range(len(self.overfit))), nan_policy='omit')
        self.rp = pearsonr(self.overfit, range(len(self.overfit)))
        logger.record(f'Spearman = {self.r.correlation:.3f}')
        logger.record(f'Pearson = {self.rp[0]:.3f}')
        self.identify_trend(self.r.correlation, self.rp[0])
        
        # decide direction
        if np.average(self.overfit)>=1:
            self.direction = 1
        else:
            self.direction = -1
        
        if self.trend != 0:
            self.calc_slope()
            self.updateMetric_Trend(logger)
            self.adjusting = False
            self.coarse_chase = 0
        else:
            self.updateMetric_xTrend(logger)

        self.record()
    
    def record(self):
        self.last_direction = self.direction
        self.last_overfit_average = np.average(self.overfit)
        
