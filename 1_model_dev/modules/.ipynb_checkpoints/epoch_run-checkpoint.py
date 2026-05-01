# modules/epoch_run.py
import traceback
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Dict
from .model import MyModel
from .train_log import LogBook, Logger
from .function_loss import Binary_Cross_Entropy, eval_classification
from .data_containers import EpochData, GradData
from .parameters import Parameters
from concurrent.futures import ThreadPoolExecutor

class BatchDataAccumulator:
    """Class to collect data for a single epoch across all batches."""
    
    def __init__(self, epoch_config=pd.DataFrame, params: Parameters):
        self.data = {}
        self.data['weight0_sum'] = 0
        self.data['batch_num'] = 0
        if epoch_config.loc['accum_loss']:
            self.data['loss'] = 0
        if epoch_config.loc['accum_metrics']:
            self.data['TP'] = 0
            self.data['FP'] = 0
            self.data['TN'] = 0
            self.data['FN'] = 0
            for key in params.metric_keys:
                self.data[key] = 0
        if epoch_config.loc['accum_fisher']:
            self.data['fisher'] = 0
        if epoch_config.loc['accum_y_pred']:
            self.data['y_pred'] = torch.tensor([], dtype=torch.float32, device='cpu')
    
    def add_batch(self, y_pred, y_true=None, weight0=1., weightp=1., weightn=1., loss: float|None=None, 
                  params: Parameters|None=None, **metrics):
        """Add data from a single batch."""
        # Convert predictions to probabilities and then to binary predictions
        # y_pred_binary = (y_pred >= 0.5).int()
        # print(y_pred.size())
        if self.phase != 'inference':
            self.batch_num += 1
            y_true = y_true.int()
            # print("a", y_true.size())
            # Initialize weights for each sample
            sample_weights = torch.ones(y_true.size(), dtype=torch.float32, device=params.DEVICE)
            # print("b", y_true.size())
            # print("c", weight0.size())
            
            # Apply sample weights (weight0)
            if not isinstance(weight0, float):
                sample_weights *= weight0
                # loss *= weight0
                self.weight0_sum += weight0.sum().item()
            else:
                self.weight0_sum += y_pred.size(0)
            # print("d", sample_weights.size())
            
            # Calculate weighted loss (weight0 can be 0)
            self.loss += loss * weight0.sum().item()
            
            # Apply class weights (weight_p for y=1, weight_n for y=0)
            class_weights = torch.where(y_true >= 0.5, weightp, weightn/params.ADJUST_N).squeeze()
            # print("e", class_weights.size())
            sample_weights *= class_weights
            # print("f", sample_weights.size())
            
            # Calculate weighted confusion matrix components
            tp_mask = (y_pred >= 0.5) & (y_true >= 0.5)
            tn_mask = (y_pred < 0.5) & (y_true < 0.5)
            fp_mask = (y_pred >= 0.5) & (y_true < 0.5)
            fn_mask = (y_pred < 0.5) & (y_true >= 0.5)
            
            # Weighted confusion matrix values
            self.batch_metrics['tp'] += (tp_mask.float() * sample_weights).sum().item()
            self.batch_metrics['tn'] += (tn_mask.float() * sample_weights).sum().item()
            self.batch_metrics['fp'] += (fp_mask.float() * sample_weights).sum().item()
            self.batch_metrics['fn'] += (fn_mask.float() * sample_weights).sum().item()
            
        if self.phase in ('test', 'inference'):
            # Ultra-fast string generation
            # self.prediction += [
            #     '; '.join([params.EC_INDEX[j] for j in range(len(params.EC_INDEX)) if y_pred[i, j]])
            #     for i in range(y_pred.size(0))
            # ]
            # print(self.prediction.size())
            # print(y_pred.size())
            # print(y_pred)
            self.prediction = torch.cat((self.prediction, y_pred.cpu()), dim=0)

    def export(self, dropout=0.0, params: Parameters|None=None) -> EpochData:
        """Export accumulated data as EpochData object."""
        
        if self.phase != 'inference':
            # Calculate average loss
            avg_loss = self.loss / self.weight0_sum
            
            # Create EpochData with loss and basic confusion matrix metrics
            epoch_data = EpochData(loss=avg_loss, dropout=dropout, params=params, metrics={})
            
            # Set confusion matrix values (these are raw counts, not rates)
            for metric in self.batch_metrics:
                epoch_data.metrics[metric] = self.batch_metrics[metric]
            if self.phase == 'test':
                epoch_data.metrics['y_pred'] = self.prediction.cpu()
            # print(epoch_data.metrics)
            return epoch_data
        else:
            return self.prediction
        
class EpochRunner:
    def __init__(self):
        # Initialize BCE loss
        self.criterion = Binary_Cross_Entropy()
        self.config = pd.read_csv(f"modules/epoch_config.csv", header=0, index_col='task').astype(bool)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=2.0**8, growth_factor=1.08)

    def switch_grad(self, train_valid_test):
        if self.config.loc['with_gradient', train_valid_test]:
            torch.autograd.set_detect_anomaly(True)
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

    def switch_dropout(self, model, train_valid_test):
        if self.config.loc['with_dropout', train_valid_test]:
            model.train()
        else:
            model.eval()
    
    def move_batch_2_device(self, params, batch):
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                batch_device[key] = {k: v.to(params.DEVICE) for k, v in value.items()}
            else:
                batch_device[key] = value.to(params.DEVICE)
        return batch_device
        
    def forward_pass(self, model, batch_device):
        # Forward pass
        with torch.amp.autocast('cuda'):
            model_output = model(batch_device['model_input'])
            if torch.isnan(model_output.y_pred).any() or torch.isinf(model_output.y_pred).any():
                print("NaN/Inf in y_pred")
                quit()
            if len(model_output.y_pred.size()) == 0:
                print(f"Failed to produce prediction")
                print(len(model_output.y_pred.size()))
                print(model_output.y_pred)
                quit()
            # print(model_output.y_pred.size())
            return model_output

    def calc_loss(self, params, batch_device, model_output):
        loss = self.criterion.calculate(
                    model_output.y_pred,
                    batch_device['evaluation_input']['y_true'],
                    batch_device['evaluation_input']['weight0'],
                    params.WEIGHT_P[train_valid_test],
                    params.WEIGHT_N[train_valid_test]
        ) / params.ACCUM_STEP
        # print("model output")
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"loss is nan or inf")
            quit()
        return loss

    def backpropagate(self, loss, model, params):
        self.scaler.scale(loss).backward() # retain_graph=True only needed when multiple backward pass is needed.
        for compo in model.optimizer:
            model._optimizer_backpropagation(scaler, compo, max_norm=params.MAX_NORM)
            if i == 0:
                print(f"Backpropagation executed")
                    
    def evaluate_grad(self, model, report_grad, grad_data):
        """Is there any nan/inf?"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"  WARNING: {name} has NaN gradients!")
                        quit()
                    if torch.isinf(param.grad).any():
                        print(f"  WARNING: {name} has Inf gradients!")
                        quit()
                        
                    if report_grad:
                        # print(param.grad[0, :5])
                        if len(param.grad.size()) == 1:
                            grad_data.add_data(epoch=epoch, name=name, grad=param.grad, params=params)
                        elif len(param.grad.size()) == 2:
                            grad_data.add_data(epoch=epoch, name=name, grad=param.grad[0], params=params)
                        
                else:
                    print(f"Param grad is None")
                    print(name)
                    quit()
        if report_grad:
            return grad_data
        return None
                    
    def run(self, train_valid_test: str, params: Parameters,
              data_source: DataLoader, model: MyModel, 
              epoch: int|None=None, grad_data:GradData|None=None, 
              logbook: LogBook|None=None, logger: Logger|None=None,) -> LogBook:
        """
        Run one epoch of training, validation, or testing.
        
        Args:
            train_valid_test: Must be 'train', 'valid', 'test', or 'inference'
            params: Parameter
            data_source: DataLoader for the data
            model: The model to run
            epoch: Current epoch number
            logbook: LogBook instance to record metrics
            logger: Logger instance for logging
        Returns:
            Updated LogBook instance
        """
        assert isinstance(params, Parameters)
        assert train_valid_test in list(self.config.columns)

        # Switch on/off grad & dropouts
        self.switch_grad(train_valid_test=train_valid_test)
        self.switch_dropout(model=model, train_valid_test=train_valid_test)

        # Move model to device
        model = model.to(params.DEVICE)

        # Progress bar
        pbar = tqdm(data_source, desc=f"Epoch {epoch} - {train_valid_test.upper()}")

        if self.config.loc[[i for i in self.config.index if i.startswith('accum')], train_valid_test].any():
            data_accumulator = BatchDataAccumulator(epoch_config=self.config, params=params)
            data_2_accum = {}
            
        for i, batch in enumerate(pbar):
            # Move batch to device
            batch_device = self.move_batch_2_device(params=params, batch=batch)
            
            # Clear gradient
            if self.config.loc['with_gradient', train_valid_test] and i % params.ACCUM_STEP == 0:
                for compo in model.optimizer:
                    model._clear_grad(compo)

            # Forward pass
            model_output = forward_pass(model=model, batch_device=batch_device)
            if self.config.loc['accum_metrics', train_valid_test]:
                data_2_accum['batch_evaluation_input'] = batch_device['evaluation_input']
                data_2_accum['y_pred'] = model_output.y_pred
                
            # Calculate loss
            if self.config.loc['calc_loss', train_valid_test]:
                loss = calc_loss(params=params, 
                                 batch_device=batch_device, 
                                 model_output=model_output)
                if self.config.loc['accum_loss', train_valid_test]:
                    data_2_accum['loss'] = loss.item()
                
            # Backward pass (only for training)
            if self.config.loc['backpropagation', train_valid_test] and \
            (i % params.ACCUM_STEP == params.ACCUM_STEP-1 or i == len(pbar)-1):
                with torch.autograd.set_detect_anomaly(True):
                    self.backpropagate(loss=loss, model=model, params=params)
                
            if i == params.ACCUM_STEP-1:
                with torch.autograd.set_detect_anomaly(True):
                    # train_valid_test decide whether to report grad
                    report_grad = i==len(pbar)-1 and self.config.loc['report_grad', train_valid_test]
                    if report_grad:
                        grad_data = self.evaluate_grad(model=model, grad_data=grad_data, \
                                                       report_grad=report_grad) 
                    else:
                        self.evaluate_grad(model=model, grad_data=grad_data, \
                                           report_grad=report_grad) 
                    self.scaler.update() # remember the scaling factors

            # Accumulate data
            if self.config.loc[[i for i in self.config.index if i.startswith('accum')], train_valid_test].any():
                data_accumulator.add_batch(data_2_accum=data_2_accum, 
                                           params=params, 
                                           config=self.config[[train_valid_test]])
                
def epoch_run(train_valid_test: str, params: Parameters,
              data_source: DataLoader, model: MyModel, 
              epoch: int|None=None, 
              logbook: LogBook|None=None, grad_data:GradData|None=None, logger: Logger|None=None,
              ) -> LogBook:
   
    for i, batch in enumerate(pbar):
        
        if train_valid_test != 'inference':
            # print(model_output.y_pred.size())
            
            if params.VERBOSE and i % 100 == 0:
                logger.record(f"Batch {i} - Loss={loss.item():.4f}")
            # print("log")
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
            })
        else:
            data_accumulator.add_batch(model_output.y_pred, params=params)
            logbook = data_accumulator.export(params=params)
            
    # ===== batch-loop ends here ==============
    if train_valid_test != 'inference':
        # Calculate metrics
        with torch.no_grad():
            epoch_data = data_accumulator.export(dropout=model.dropout_tuner['FFN'].metric, params=params) # add 
            epoch_data = eval_classification(
                epoch_data, 
                return_metrics=True
            )
        logbook.write_epoch_data(epoch, train_valid_test, epoch_data, 
                                 params, logger)

    if train_valid_test == 'valid':
        return logbook
    elif train_valid_test == 'train':
        return logbook, grad_data
    elif train_valid_test == 'test':
        return logbook, epoch_data

def eval_run(params: Parameters, data_source: DataLoader, 
               model: MyModel, logger: Logger|None=None,
              ) -> LogBook:
    """
    Run one epoch of training w/o updating model parameters.
    
    Args:
        data_source: DataLoader for the data
        model: The model to run
        logger: Logger instance for logging
        params: all paramters
    Returns:
        Updated LogBook instance
    """
    
    # Progress bar
    pbar = tqdm(data_source, desc=f"Fisher Information")
    fisher_dict = {}
    for name, _ in model.named_parameters():
        fisher_dict[name] = 0
    
    for i, batch in enumerate(pbar):
        # Backward pass (only for training)
        if i % params.ACCUM_STEP == params.ACCUM_STEP-1 or i == len(pbar)-1:
            with torch.autograd.set_detect_anomaly(True):
                # Construct gradient graph
                scaler.scale(loss).backward() # retain_graph=True only needed when multiple backward pass is needed.
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.data ** 2
                
    # Average over batches
    fisher_dict = {k:torch.sqrt((v.detach()/len(pbar))).cpu() for k, v in fisher_dict.items()}
    return fisher_dict
    