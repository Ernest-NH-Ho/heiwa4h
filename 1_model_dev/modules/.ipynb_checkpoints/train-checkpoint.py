import torch
from tqdm.auto import tqdm
import numpy as np
from modules.function_loss import MAE_Soft, calc_SSR, calc_SST # Binary_Cross_Entropy, eval_classification
from modules.plot import plotHexbin
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional, Dict

@dataclass
class RegressionEvaluationOutput:
    """Everything should be in cpu at this stage after training/valid"""
    loss : Optional[Dict] = field(default=None)
    # adjR2 : Optional[Dict] = field(default=None)
    # expert_count : Optional[Dict] = field(default=None)
    
class DataAccumulator_WithinEpoch_Regression:
    def __init__(self, data_target_list:list, train_valid_test='train', num_expert=27, scatterplot=False):
        # basic metrics
        self.n = 0
        self.weight0 = 0  # sum of weight0
        self.train_valid_test = train_valid_test

        # target-specific metrics
        self.target_metrics = {}
        self.target_metrics['loss'] = 0
        
    def add_basic_metrics(self, batch):
        # self.n += dataset_output['evaluation_input']['length'].sum().item()
        # self.weight0 = (batch['evaluation_input']['weight0'] * dataset_output['evaluation_input']['length']).sum().item()
        self.n += len(batch)
        self.weight0 = batch['evaluation_input']['weight0'].sum().item()

    def add_target_specific_metrics(self, model_output:'ModelOutput', dataset_output:dict, loss):
        self.target_metrics[data_target]['loss'] += loss.sum().item()
        
        # ==== Monitor weighted expert counts ======
        # expert_count= (model_output.expert_count * dataset_output['evaluation_input']['weight0'].unsqueeze(-1)).sum(dim=0).cpu()
        # # print(f"\nExpert Employment in this batch:\n{expert_count}\n")

        # if update_expert_bias:
        #     self.target_metrics[data_target]['expert_count_tmp'] += expert_count
        #     self.target_metrics[data_target]['expert_count'] += self.target_metrics[data_target]['expert_count_tmp']
        # else:
        #     self.target_metrics[data_target]['expert_count'] += expert_count
        
        # If one expert was newly added or # Add expert weighted usage counts
        # n = len(self.target_metrics[data_target]['expert_count'])
        # if n==0:
        #     self.target_metrics[data_target]['expert_count'] = torch.cat([self.target_metrics[data_target]['expert_count'], expert_count], 
        #                                                                  dim=0)
        # else:
        #     self.target_metrics[data_target]['expert_count'] += expert_count

    # def calc_adjR2(self):
    #     """all y are weighted by w0 already at add_target_specific_metrics"""
    #     for data_target in self.data_target_list:
    #         ssr = calc_SSR(self.target_metrics[data_target]['y_pred'], 
    #                        self.target_metrics[data_target]['y_true'], target=data_target)
    #         sst = calc_SST(self.target_metrics[data_target]['y_true'], 
    #                        train_valid_test=self.train_valid_test, target=data_target)
    #         assert ssr > 0
    #         assert sst > 0
    #         assert self.weight0 > 0
    #         self.target_metrics[data_target]['adjR2'] = 1 - (ssr * (self.denom - 1))/(sst * (self.denom - 2))

    # def plot_hexbin(self, hp_folder, epoch, train_valid_test):
    #     for data_target in self.data_target_list:
    #         self.target_metrics[data_target]['y_pred'] = self.target_metrics[data_target]['y_pred'].detach().cpu().numpy()
    #         self.target_metrics[data_target]['y_true'] = self.target_metrics[data_target]['y_true'].detach().cpu().numpy()
    #         if data_target == 'mut':
    #             axis_label = 'Mutation'
    #         elif data_target == 'epi':
    #             axis_label = 'Epistasis'
    #         plotHexbin(self.target_metrics[data_target]['y_true'], 
    #                     self.target_metrics[data_target]['y_pred'], 
    #                     f"{axis_label} Score", 
    #                     f"model/{hp_folder}/scatter_{data_target}_{train_valid_test}")

    def generate_output(self):
        # self.calc_adjR2()
        # if hasattr(self, 'weight0'):
        #     denom = self.weight0
        # else:
        #     denom = self.n 
        # denom *= self.seq_len
        # for data_target in self.data_target_list:
            # self.target_metrics[data_target]['loss'] /= self.denom
            # self.target_metrics[data_target]['expert_count'] /= self.denom
        
        ## Prepare output here
        output = {}
        for data_target in self.data_target_list:
            output[data_target] = RegressionEvaluationOutput(loss=self.target_metrics['loss'])
            # if 'adjR2' in self.target_metrics[data_target]:
            #     output[data_target].adjR2 = self.target_metrics[data_target]['adjR2']
            # if 'expert_count' in self.target_metrics[data_target]:
            #     output[data_target].expert_count = self.target_metrics[data_target]['expert_count'].numpy()
        return output
            

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj  # Leave non-tensors untouched
        
def epoch_run(train_valid_test, \
              data_source, model, \
              epoch, logbook, logger, \
              hp_folder, verbose=False, device='cuda'): # weight_1
    # train_source, model, model_component_config, data_label_config, layer_2_update, data_target, model_target, logger
    """called in each epoch"""
    """Designed for regression currently """
    
    if train_valid_test == 'train':
        model.train()
        torch.autograd.set_detect_anomaly(True)  # use to identify source of inf
        scaler = torch.cuda.amp.GradScaler()
        accumulation_steps = 4
        gradient_mode = True
        scatterplot=False
    else:
        model.eval()
        gradient_mode = False

    # with gradient_switcher(gradient_mode): 
    with torch.set_grad_enabled(gradient_mode):
        # least_expert_count = 0
        tot_batch = len(data_source)
        data_accumulator = DataAccumulator_WithinEpoch_Regression(scatterplot=scatterplot, 
                                                                  data_target_list=data_target_list, 
                                                                  train_valid_test=train_valid_test)

        ## ====================== main process starts here ======================================
        data_source.data_target = data_target_list
        for i, batch in enumerate(tqdm(data_source)):
            torch.cuda.empty_cache()
            batch['model_input'] = move_to_device(batch['model_input'], device)
            batch['evaluation_input']['weight0'] = move_to_device(batch['evaluation_input']['weight0'], device)
            
            ## For MoE
            
            # update_expert_bias = False
            # if train_valid_test == 'train':
            #     if i < 12:
            #         if i%2 == 1:
            #             update_expert_bias = True
            #     else:
            #         if i%6 == 5:
            #             update_expert_bias = True
                    
                

            ## Reset gradient
            if train_valid_test == 'train':
                for compo in model.optimizer:
                    if train_schedule._content.loc[epoch, f"grad_{compo}"]==True \
                    or compo in data_target_list:
                        model.optimizer[compo].zero_grad()
                        
            ##  Change seeds  
            # for compo in model.component:
            #     if compo != 'bert':
            #         model.component[compo]['MoE'].gate.seed_add_one()
                    
            model_output = model(batch['model_input'], data_target)

            batch['evaluation_input']['y_true'][data_target] = move_to_device(batch['evaluation_input']['y_true'][data_target], device)
            
            ## Safety against gradient explosion
            # if train_valid_test == 'train':
            #     if model_output.y_pred.max().item() > 15:
            #         raise ValueError(f"Model predicting values over 20")
            #     if model_output.y_pred.min().item() < -10:
            #         raise ValueError(f"Model predicting values smaller than 15")

            # if i % 500 == 99:
            #     print(f"At epoch {epoch+1} batch {i+1} for {data_target}")
            #     print(f"Cumulative expert count at epoch {epoch+1} batch {i+1}:\n{data_accumulator.target_metrics[data_target]['expert_count']}")
            #     print(f"MoE router bias:\n{model.component[data_target]['MoE'].gate.linear.bias}")
            
            ## Masking
            # while len(batch['model_input']['attention_mask'].size()) < len(batch['evaluation_input']['y_true'][data_target].size()):
            #     batch['model_input']['attention_mask'] = batch['model_input']['attention_mask'].unsqueeze(-1)
            # # print(f"y_true: {batch['evaluation_input']['y_true'][data_target].size()}")
            # # print(f"attention_mask: {batch['model_input']['attention_mask'].size()}")
            # # print(f"disorder_mask: {batch['model_input']['disorder_mask'].size()}")
            # batch['evaluation_input']['y_true'][data_target] = batch['evaluation_input']['y_true'][data_target] * batch['model_input']['attention_mask']
            # batch['evaluation_input']['y_true'][data_target] = batch['evaluation_input']['y_true'][data_target] * batch['model_input']['disorder_mask']
           
            # Calc loss --> Scaled loss backward
            loss = MAE_Soft.calculate(model_output.y_pred, batch['evaluation_input']['y_true'], 
                                      batch['evaluation_input']['weight0'])

            if i %500 == 99:
                print(f"loss = {loss.item()}")
                print("loss.requires_grad:", loss.requires_grad)
                print("loss.grad_fn:", loss.grad_fn)
                print("y_pred.requires_grad:", model_output.y_pred.requires_grad)
                print("y_pred.grad_fn:", model_output.y_pred.grad_fn)
                # print("Any param requires_grad:", any(p.requires_grad for p in model.parameters()))
                # print("Any param has grad:", any(p.grad is not None for p in model.parameters()))
            
            # Backpropagation with orthogonal loss
            if train_valid_test == 'train':
                with torch.autograd.set_detect_anomaly(True):
                    scaler.scale(loss).backward(retain_graph=True)  # may scale to float16 or float64

                    # Orthogonal loss
                    # if i == tot_batch-1:
                    #     report_ortho = True
                    #     orthogonal_loss_report = model.calc_ortho_loss(logger, train_valid_test, report=False) #######
                    # else:
                    #     report_ortho = False
                    #     model.calc_ortho_loss(logger, train_valid_test, report=report_ortho)

                    # Backpropagation
                    for data_target in model.optimizer:
                        if train_schedule._content.loc[epoch, f"grad_{data_target}"]==True:
                            model.optimizer_backpropagation(scaler, data_target, max_norm=1.0)
            else:
                report_ortho = False
            
            ## Accumkulate target-specific metrics
            data_accumulator.add_target_specific_metrics(model_output, batch, loss)
            
            ## =========== one for loop finished ============

            # if i%500 == 99:
            #     print(f"model output max: {model_output.y_pred.max().item()}")
            #     print(f"model output mean: {model_output.y_pred.mean().item()}")
            #     print(f"model output min: {model_output.y_pred.min().item()}")
            #     print(f"model output:\n{model_output.y_pred[:4, :4]}")
                

            ## Accumulate batch_metrics: n, weight0
            data_accumulator.add_basic_metrics(batch)

            ## Update MoE bias
            # if train_valid_test == 'train':
            #     if i % 11:
            #         # How to trim the bias term in phase 1
            #         if epoch == 0 and i < 100:
            #             model.component[data_target]['MoE'].gate.multiply_bias(0.8) # half the bias terms
    
            #     if epoch > 0 or i > 100:
            #         # How to add bias term in phase 2
            #         if update_expert_bias:
            #             for data_target in data_target_list:
            #                 expert_count_tmp = data_accumulator.target_metrics[data_target]['expert_count_tmp'] / data_accumulator.target_metrics[data_target]['expert_count_tmp'].sum()
            #                 expert_count = data_accumulator.target_metrics[data_target]['expert_count'] / data_accumulator.target_metrics[data_target]['expert_count'].sum()
            #                 expert_count -= expert_count_tmp
    
            #                 _, j = torch.topk(expert_count, k=3, dim=0, largest=True, sorted=True)
            #                 model.component[data_target]['MoE'].gate.change_bias(j, -0.01)
            #                 _, j = torch.topk(expert_count, k=7, dim=0, largest=False, sorted=True)
            #                 model.component[data_target]['MoE'].gate.change_bias(j, 0.01)
    
            #                 j = (expert_count_tmp < 0.001).nonzero(as_tuple=False).squeeze()
            #                 model.component[data_target]['MoE'].gate.change_bias(j, 0.01)
            #                 expert_count_tmp /= expert_count_tmp.min() + 1e-7
            #                 j = (expert_count_tmp > 1000).nonzero(as_tuple=False).squeeze()
            #                 model.component[data_target]['MoE'].gate.change_bias(j, -0.01)
            #                 data_accumulator.target_metrics[data_target]['expert_count_tmp'] = torch.zeros((data_accumulator.num_expert), \
            #                                                                                                device='cpu', dtype=torch.float32)
            #     else:
            #         # How to add bias term in phase 1
            #         if update_expert_bias:
            #             _, j = torch.topk(data_accumulator.target_metrics[data_target]['expert_count'], k=3, dim=0, largest=True, sorted=True)
            #             # j = torch.argmax(data_accumulator.target_metrics[data_target]['expert_count']).tolist()
            #             model.component[data_target]['MoE'].gate.change_bias(j, -0.02)
            #             _, j = torch.topk(data_accumulator.target_metrics[data_target]['expert_count'], k=7, dim=0, largest=False, sorted=True)
            #             # j = torch.argmin(data_accumulator.target_metrics[data_target]['expert_count']).tolist()
            #             model.component[data_target]['MoE'].gate.change_bias(j, 0.03)
            #             data_accumulator.target_metrics[data_target]['expert_count_tmp'] = torch.zeros((data_accumulator.num_expert), \
            #                                                                                            device='cpu', dtype=torch.float32)
    
                
            #     ## Update learing rate until 1e-7 only in epoch 1
            #     if train_valid_test == 'train' and epoch == 0:
            #         for data_target in data_target_list:
            #             lr = model.optimizer[data_target].param_groups[0]["lr"]
            #             if lr < 2e-7:
            #                 model.change_lr(data_target, lr+2e-9)
            #         if i%500 == 99:
            #             print(f"Learning rate for {data_target} = {lr}")
                            
        ## All for-loop ends here
        
        ## =============== Main process ends ================================
        # Write log
        if train_valid_test != 'valid':
            model = None
        if report_ortho:
            logbook.add_epoch_output(epoch=epoch, train_valid_test=train_valid_test, 
                                     epoch_run_output=data_accumulator.generate_output(), \
                                     model=model)
        else:
            logbook.add_epoch_output(epoch=epoch, train_valid_test=train_valid_test, 
                                     epoch_run_output=data_accumulator.generate_output(), model=model)
        logbook.write_epoch_log(epoch=epoch, train_valid_test=train_valid_test, 
                                logger=logger)

        if scatterplot:
            data_accumulator.plot_hexbin(hp_folder, epoch, train_valid_test)

    return logbook
        
