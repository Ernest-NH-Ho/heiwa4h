import torch
import pandas as pd
from modules.train_control.parameters import Parameters
from modules.data_containers import EpochData
import numpy as np

class BatchDataAccumulator:
    """Class to collect data for a single epoch across all batches."""
    
    def __init__(self, epoch_config:pd.DataFrame, train_valid_test:str, params: Parameters):
        self.data = {}
        self.train_valid_test = train_valid_test
        self.data['weight0_sum'] = 0
        if epoch_config.loc['accum_loss', train_valid_test]:
            self.data['loss'] = 0
        if epoch_config.loc['accum_metrics', train_valid_test]:
            self.data['tp'] = 0
            self.data['fp'] = 0
            self.data['tn'] = 0
            self.data['fn'] = 0
            for key in params.metric_keys:
                self.data[key] = 0
        if epoch_config.loc['accum_fisher', train_valid_test]:
            self.data['fisher'] = {}
        if epoch_config.loc['accum_y_pred', train_valid_test]:
            self.data['y_pred'] = torch.tensor([], dtype=torch.float32, device='cpu')
        if epoch_config.loc['accum_y_true_pred', train_valid_test]:
            self.data['y_pred'] = torch.tensor([], dtype=torch.float32, device='cpu')
            self.data['y_true'] = torch.tensor([], dtype=torch.float32, device='cpu')
        if params.save_intermediate_layer:
            self.data['intermediate_output'] = {}
        
    def add_batch(self, epoch_config, data_2_accum:dict, params:Parameters, logger=None):
        """ loss = is already weighted by weight_0 
        tp tn fp fn is not weighted by weight_0 """
        if params.DEBUG:
            logger.record(f"Batch accumulator added weight0 to weight0_sum")
            
        if epoch_config.loc['accum_loss', self.train_valid_test] or epoch_config.loc['accum_metrics', self.train_valid_test]:
            if isinstance(data_2_accum['weight0'], torch.Tensor):
                weight0 = data_2_accum['weight0']
                w0_sum = weight0.sum()
                self.data['weight0_sum'] += w0_sum
                if params.DEBUG:
                    logger.record(f"batch weight0_sum {weight0.sum().item()}")
                    logger.record(f"cumulative weight0_sum {self.data['weight0_sum']}")
            else:
                w0_sum = data_2_accum['y_pred'].numel()
                self.data['weight0_sum'] += w0_sum
                weight0 = 1

        # Loss is weight0-adjusted at its calculation
        if epoch_config.loc['accum_loss', self.train_valid_test]:
            if params.DEBUG:
                logger.record(f"Batch sum of loss = {data_2_accum['loss']}")
                logger.record(f"Weighted batch sum of loss = {data_2_accum['loss']  * w0_sum}")
            self.data['loss'] += (data_2_accum['loss'] * w0_sum)
            if isinstance(self.data['loss'], torch.Tensor):
                self.data['loss'] = self.data['loss'].item() # get sum(loss * weight_0)
            
        if epoch_config.loc['accum_metrics', self.train_valid_test]:
            if params.DEBUG:
                logger.record(f"Batch accumulator added metrics weighted by weight0, weightp, weightn")
                
            # Calculate sample_weights
            class_weights = torch.where(data_2_accum['y_true'] >= 0.5, \
                                        params.WEIGHT_P[self.train_valid_test], \
                                        params.WEIGHT_N[self.train_valid_test]).squeeze()
            if params.DEBUG:
                logger.record(f"size of y_true {data_2_accum['y_true'].size()}")
                logger.record(f"size of class weights {class_weights.size()}")
            sample_weights = torch.ones(data_2_accum['y_true'].size(), dtype=torch.float32, device=params.DEVICE) * weight0
            sample_weights *= class_weights
            if params.DEBUG:
                logger.record(f"size of the sample weight {sample_weights.size()}")
            # quit()
            
            # Calculate weighted confusion matrix components
            if params.DEBUG:
                logger.record(f"Batch accumulator constructed tp, tn, fp, fn")
                # print("mask", data_2_accum["mask"])
                # quit()
            if "mask" in data_2_accum:
                tp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] >= 0.5) & data_2_accum["mask"]
                tn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] < 0.5) & data_2_accum["mask"]
                fp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] < 0.5) & data_2_accum["mask"]
                fn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] >= 0.5) & data_2_accum["mask"]
                if params.DEBUG:
                    logger.record(f"{data_2_accum['mask']}")
            elif "mask_A" in data_2_accum and "mask_B" in data_2_accum:
                # Need to generatlize to ABCDE
                # print("HI")
                tp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] >= 0.5) & data_2_accum["mask_A"].unsqueeze(-1) & data_2_accum['mask_B'].unsqueeze(-2)
                tn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] < 0.5)  & data_2_accum["mask_A"].unsqueeze(-1) & data_2_accum['mask_B'].unsqueeze(-2)
                fp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] < 0.5) & data_2_accum["mask_A"].unsqueeze(-1) & data_2_accum['mask_B'].unsqueeze(-2)
                fn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] >= 0.5) & data_2_accum["mask_A"].unsqueeze(-1) & data_2_accum['mask_B'].unsqueeze(-2)
            else:
                tp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] >= 0.5)
                tn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] < 0.5)
                fp_mask = (data_2_accum['y_pred'] >= 0.5) & (data_2_accum['y_true'] < 0.5)
                fn_mask = (data_2_accum['y_pred'] < 0.5) & (data_2_accum['y_true'] >= 0.5)
                if params.DEBUG:
                    print("y_pred", data_2_accum['y_pred'])
                    print("y_true", data_2_accum['y_true'])

            if params.DEBUG:
                logger.record(f"Batch accumulator unweighted tp, tn, fp, fn")
                logger.record(f"tp {tp_mask.float().sum().item()}")
                logger.record(f"tn {tn_mask.float().sum().item()}")
                logger.record(f"fp {fp_mask.float().sum().item()}")
                logger.record(f"fn {fn_mask.float().sum().item()}")
                # quit()
            
            # Weighted confusion matrix values
            self.data['tp'] += (tp_mask.float() * sample_weights).sum().item()
            self.data['tn'] += (tn_mask.float() * sample_weights).sum().item()
            self.data['fp'] += (fp_mask.float() * sample_weights).sum().item()
            self.data['fn'] += (fn_mask.float() * sample_weights).sum().item()
            if params.DEBUG:
                logger.record(f"Batch accumulator weighted tp, tn, fp, fn")
                logger.record(f"tp {(tp_mask.float() * sample_weights).sum().item()}")
                logger.record(f"tn {(tn_mask.float() * sample_weights).sum().item()}")
                logger.record(f"fp {(fp_mask.float() * sample_weights).sum().item()}")
                logger.record(f"fn {(fn_mask.float() * sample_weights).sum().item()}")
                logger.record(f"Batch accumulator cumulative weighted tp, tn, fp, fn")
                logger.record(f"tp {self.data['tp']}")
                logger.record(f"tn {self.data['tn']}")
                logger.record(f"fp {self.data['fp']}")
                logger.record(f"fn {self.data['fn']}")
                # quit()
            
        if epoch_config.loc['accum_y_true_pred', self.train_valid_test]:
            self.data['y_pred'] = torch.cat((self.data['y_pred'], data_2_accum['y_pred'].cpu()), dim=0)
            self.data['y_true'] = torch.cat((self.data['y_true'], data_2_accum['y_true'].cpu()), dim=0)
            if params.DEBUG:
                logger.record(f"Batch accumulator added all y_pred and y_true")
            
        elif epoch_config.loc['accum_y_pred', self.train_valid_test]:
            self.data['y_pred'] = torch.cat((self.data['y_pred'], data_2_accum['y_pred'].cpu()), dim=0)
            if params.DEBUG:
                logger.record(f"Batch accumulator added all y_pred")
            
        if epoch_config.loc['accum_fisher', self.train_valid_test]:
            for name, param in data_2_accum['model_named_parameters']:
                if param.grad is not None and 'weight' in name:
                    n = param.grad.size(-1)
                    if n not in self.data['fisher']:
                        self.data['fisher'][n] = {}
                    self.data['fisher'][n][name] = self.data['fisher'][n].get(name, 0) + param.grad.data**2

        if params.save_intermediate_layer:
            if params.DEBUG:
                logger.record(f"Batch accumulator added all intermediate_output")
            # print(data_2_accum["intermediate_output"])
            for k in data_2_accum["intermediate_output"]:
                if not k in self.data['intermediate_output']:
                    self.data['intermediate_output'][k] = torch.tensor([], dtype=torch.float32, device='cpu')
                self.data['intermediate_output'][k] = torch.cat((self.data['intermediate_output'][k], data_2_accum['intermediate_output'][k].cpu()), dim=0)

    def save_data(self, epoch_config:pd.DataFrame, params:Parameters, train_valid_test:str):
        """ snapshots do not require accumulation """
        if epoch_config.loc['save_fisher', self.train_valid_test]:
            for n in self.data['fisher']:
                self.data['fisher'][n] = {name: (grad.mean(dim=0) if len(grad.size()) > 1 else grad) \
                                          for name, grad in self.data['fisher'][n].items()}
                self.data['fisher'][n] = {name:grad.detach().cpu().numpy() \
                                          for name, grad in self.data['fisher'][n].items()}
                self.data['fisher'][n] = pd.DataFrame.from_dict(self.data['fisher'][n])
                self.data['fisher'][n].to_csv(f"{params.DIRECTORIES.LOG_DIR}/fisher_info_len{n}.csv", index=False)
            
        if epoch_config.loc['save_y_pred', self.train_valid_test]:
            if hasattr(params, 'LABEL_MAP'):
                output = pd.DataFrame()
                indices = torch.argwhere(torch.where(self.data['y_pred']>=0.5, 1, 0).bool())
                n = self.data['y_pred'].size(0)
                output["y_pred"] = [[] for _ in range(n)]
                output["logit"] = [[] for _ in range(n)]
                for (i0, i1) in indices:
                    i0 = i0.item()
                    i1 = i1.item()
                    output.loc[i0, "y_pred"].append(params.LABEL_MAP[i1])
                    output.loc[i0, "logit"].append(str(self.data['y_pred'][i0, i1].item()))
                output["y_pred"] = output["y_pred"].apply(lambda x:'; '.join(x))
                output["logit"] = output["logit"].apply(lambda x:'; '.join(x))
                output.to_csv(f"{params.DIRECTORIES.LOG_DIR}/y_pred_{self.train_valid_test}.csv", index=False)
            else:
                y_pred = self.data['y_pred'].cpu().numpy()
                np.save(f"{params.DIRECTORIES.LOG_DIR}/y_pred_{self.train_valid_test}.npy", y_pred)
            
        if epoch_config.loc['save_y_true_pred', self.train_valid_test]:
            if hasattr(params, 'LABEL_MAP'):
                output = pd.DataFrame()
                indices = torch.argwhere(torch.where(self.data['y_pred']>=0.5, 1, 0).bool())
                n = self.data['y_pred'].size(0)
                output["y_pred"] = [[] for _ in range(n)]
                output["logit"] = [[] for _ in range(n)]
                for (i0, i1) in indices:
                    i0 = i0.item()
                    i1 = i1.item()
                    output.loc[i0, "y_pred"].append(params.LABEL_MAP[i1])
                    output.loc[i0, "logit"].append(str(self.data['y_pred'][i0, i1].item()))
                output["y_pred"] = output["y_pred"].apply(lambda x:'; '.join(x))
                output["logit"] = output["logit"].apply(lambda x:'; '.join(x))

                indices = torch.argwhere(torch.where(self.data['y_true']>=0.5, 1, 0).bool())
                n = self.data['y_true'].size(0)
                output["y_true"] = [[] for _ in range(n)]
                for (i0, i1) in indices:
                    i0 = i0.item()
                    i1 = i1.item()
                    output.loc[i0, "y_true"].append(params.LABEL_MAP[i1])
                output["y_true"] = output["y_true"].apply(lambda x:'; '.join(x))
                output.to_csv(f"{params.DIRECTORIES.LOG_DIR}/y_true_pred_{self.train_valid_test}.csv", index=False)
            else:
                y_true = self.data['y_true'].cpu().numpy()
                y_pred = self.data['y_pred'].cpu().numpy()
                np.save(f"{params.DIRECTORIES.LOG_DIR}/y_true_{self.train_valid_test}.npy", y_true)
                np.save(f"{params.DIRECTORIES.LOG_DIR}/y_pred_{self.train_valid_test}.npy", y_pred)

        if params.save_intermediate_layer:
            for k in self.data["intermediate_output"]:
                print(k, self.data["intermediate_output"][k].max())
                torch.save(self.data["intermediate_output"][k], f"{params.DIRECTORIES.LOG_DIR}/{k}.pt")

    def calc_mean_from_sum(self, epoch_config:pd.DataFrame, train_valid_test:str):
        if epoch_config.loc['accum_loss', self.train_valid_test]:
            print(f"weighted sum of loss = {self.data['loss']}")
            self.data['loss'] /= self.data['weight0_sum']
            if isinstance(self.data['loss'], torch.Tensor):
                self.data['loss'] = self.data['loss'].item()
            print(f"weighted mean of loss = {self.data['loss']}")

    def export(self, params: Parameters, epoch_config:pd.DataFrame, 
               train_valid_test:str) -> EpochData:
        """Export accumulated data as EpochData object."""
        self.calc_mean_from_sum(epoch_config=epoch_config, train_valid_test=self.train_valid_test)

        if params.DEBUG:
            logger.record(f"tp_sum = {self.data['tp']}")
            logger.record(f"tn_sum = {self.data['tn']}")
            logger.record(f"fp_sum = {self.data['fp']}")
            logger.record(f"fn_sum = {self.data['fn']}")
        avg_loss = None
        metrics = {}
        if epoch_config.loc['log_loss', self.train_valid_test]:
            avg_loss = self.data['loss']
        if epoch_config.loc['log_metrics', self.train_valid_test]:
            for metric in params.metric_keys:
                metrics[metric] = self.data[metric]
                
        # Create & return EpochData with loss and basic confusion matrix metrics
        # print(dropout)
        return EpochData(loss=avg_loss, metrics=metrics)
        