import os
import traceback
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.model.model import MyModel
from modules.train_log import LogBook, Logger
from modules.functions.loss import Binary_Cross_Entropy, eval_classification
from modules.epoch_control.batch_accumulator import BatchDataAccumulator
from modules.train_control.parameters import Parameters
from modules.train_control.computation_monitor import ComputationMonitor

class EpochRunner:
    def __init__(self, params:Parameters,  expected_tasks:list=['train', 'valid', 'test']):
        # Initialize BCE loss
        self.config = pd.read_csv(f"modules/epoch_control/epoch_config.csv", header=0, index_col='task').astype(bool)
        # print(self.config)
        self.accum_factor = {"train": 1, "valid": 1, "test": 1, "fisher":1} # batch size of valid is automatically 2x of train
        self.computation_monitor = ComputationMonitor(params=params)
        if self.config.loc['calc_loss', expected_tasks].any():
            self.criterion = Binary_Cross_Entropy()
        if self.config.loc['with_gradient', expected_tasks].any():
            self.scaler = torch.amp.GradScaler('cuda', init_scale=64, growth_factor=1.08)
        if self.config.loc['save_grad_snapshot', expected_tasks].any():
            if os.path.exists(f"{params.DIRECTORIES.LOG_DIR}/grad_snapshot.csv"):
                self.grad_data_snapshot = pd.read_csv(f"{params.DIRECTORIES.LOG_DIR}/grad_snapshot.csv", header=0)
            else:
                self.grad_data_snapshot = pd.DataFrame()
        # if hasattr(params, "save_intermediate_layer"):
        #     self.save_intermediate_layer = params.save_intermediate_layer
        #     if params.save_intermediate_layer:
        #         self.intermediate_output = {}

    def require_grad(self, train_valid_test, logger):
        if self.config.loc['with_gradient', train_valid_test]:
            torch.autograd.set_detect_anomaly(True)
            torch.set_grad_enabled(True)
            logger.record(f"Gradient turned on")
        else:
            torch.set_grad_enabled(False)
            logger.record(f"Gradient turned off")

    def require_dropout(self, model, train_valid_test, logger):
        if self.config.loc['with_dropout', train_valid_test]:
            model.train()
            logger.record(f"model.train() called")
        else:
            model.eval()
            logger.record(f"model.eval() called")
    
    def move_batch_2_device(self, params, batch):
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                batch_device[key] = {k: v.to(params.DEVICE) for k, v in value.items()}
            else:
                batch_device[key] = value.to(params.DEVICE)
        return batch_device
        
    def forward_pass(self, model, batch_device, log_all_tensors=False):
        # Forward pass
        with torch.amp.autocast('cuda'):
            model_output = model(batch_device['model_input'], log_all_tensors)
            # print(model_output.y_pred)
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

    def calc_loss(self, params, train_valid_test, Wpn_idx, batch_device, model_output):
        L = None
        if 'L' in batch_device['evaluation_input']:
            # print("There is L")
            # quit()
            L = batch_device['evaluation_input']['L']
        loss = self.criterion.calculate(
                    y_pred=model_output.y_pred,
                    y_true=batch_device['evaluation_input']['Y'],
                    weight_0=batch_device['evaluation_input'].get('W0', 1),
                    weight_p=params.WEIGHT_P[Wpn_idx],
                    weight_n=params.WEIGHT_N[Wpn_idx], 
                    L=L,
        ) / (params.ACCUM_STEP * self.accum_factor[train_valid_test])
        # assert ~(torch.isnan(loss).any())
        # assert ~(torch.isinf(loss).any())
        # assert (loss.abs() < 1000).all()
        # print(batch_device['evaluation_input']['Y'])
        # print(batch_device['evaluation_input'].get('W0', 1))
        return loss

    def _update_model_params(self, model, params, epoch):
        pass

    def evaluate_grad(self, model, params, train_valid_test, save_grad_snapshot:bool, epoch:int|None=None):
        """Is there any nan/inf?"""
        if save_grad_snapshot:
            n = len(self.grad_data_snapshot)
            self.grad_data_snapshot.loc[n, "epoch"] = epoch
            self.grad_data_snapshot.loc[n, "train_valid_test"] = train_valid_test
        for name, param in model.named_parameters():
            # print(name, param.grad)
            if param.requires_grad and 'weight' in name:
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"  WARNING: {name} has NaN gradients!")
                        quit()
                    if torch.isinf(param.grad).any():
                        print(f"  WARNING: {name} has Inf gradients!")
                        quit()
                        
                    if save_grad_snapshot:
                        # print(param.grad[0, :5])
                        if len(param.grad.size()) == 1:
                            idx = min(len(param.grad), params.GRAD_RECORD_DIM)
                            self.grad_data_snapshot.loc[n, [f'{name}_{i}' for i in range(idx)]] = param.grad[:idx].cpu().tolist()
                        elif len(param.grad.size()) == 2:
                            idx = min(len(param.grad[0]), params.GRAD_RECORD_DIM)
                            self.grad_data_snapshot.loc[n, [f'{name}[0]_{i}' for i in range(idx)]] = param.grad[0, :idx].cpu().tolist()
                else:
                    print(f"{name} grad is None")
                    print(name)
                    quit()
        # quit()
        if save_grad_snapshot:
            self.grad_data_snapshot.to_csv(f"{params.DIRECTORIES.LOG_DIR}/grad_snapshot.csv", index=False)
                    
    def run(self, train_valid_test: str, params: Parameters,
            data_source: DataLoader, model: MyModel, 
            epoch: int|None=None, 
            logbook: LogBook|None=None, logger: Logger|None=None,
            computation_monitor:bool=False
           ) -> LogBook:
        """         Run one epoch of training, validation, or testing.
        
        Args:
            train_valid_test: Must be 'train', 'valid', 'test', or 'inference'
            params: Parameter
            data_source: DataLoader for the data
            model: The model to run
            epoch: Current epoch number
            logbook: LogBook instance to record metrics
            logger: Logger instance for logging
        Returns:
            Updated LogBook instance          """
        assert isinstance(params, Parameters)
        assert train_valid_test in list(self.config.columns)
        if train_valid_test == 'fisher':
            Wpn_idx = 'train'
        else:
            Wpn_idx = train_valid_test

        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        
        # Switch on/off grad & dropouts
        self.require_grad(train_valid_test=train_valid_test, logger=logger)
        self.require_dropout(model=model, train_valid_test=train_valid_test, logger=logger)

        # Move model to device
        model = model.to(params.DEVICE)

        # Start computation monitor
        if computation_monitor:
            self.computation_monitor.register_model(model=model, logger=logger)

        # Progress bar
        logger.record(f"Creating pbar")
        pbar = tqdm(data_source, desc=f"Epoch {epoch} - {train_valid_test.upper()}")

        if self.config.loc[[i for i in self.config.index if i.startswith('accum')], train_valid_test].any() or params.save_intermediate_layer:
            logger.record(f"Initializing batch accumulator")
            data_accumulator = BatchDataAccumulator(epoch_config=self.config, 
                                                    train_valid_test=train_valid_test, params=params)
            
        for i, batch in enumerate(pbar):
            # Initialize data accumulator, if required
            if self.config.loc[[i for i in self.config.index if i.startswith('accum')], train_valid_test].any():
                data_2_accum = {}

            # Computation monitor
            if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                logger.record(f"computation_monitor start")
                self.computation_monitor.start()
            
            # Move batch to device
            batch_device = self.move_batch_2_device(params=params, batch=batch)
            if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                self.computation_monitor.checkpoint(name="1_move_to_device")
                
            # Clear gradient
            if self.config.loc['model_parm_update', train_valid_test] and i % params.ACCUM_STEP == 0:
                model._clear_grad(epoch)
                if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                    self.computation_monitor.checkpoint(name="2_clear_grad")

            # ================ Forward pass =======================
            log_all_tensors = False
            if computation_monitor and i%self.computation_monitor._freq_batch==0:
                log_all_tensors = True
            model_output = self.forward_pass(model=model, batch_device=batch_device, 
                                             log_all_tensors=log_all_tensors)

            # Accumulate data for evaluation
            if self.config.loc['accum_metrics', train_valid_test]:
                data_2_accum['batch_evaluation_input'] = batch_device['evaluation_input']
                if params.DEBUG:
                    logger.record(f"Epoch Runner registered batch evaluation input for accumulation")

            # Accumulate y_pred y_true
            if self.config.loc[['accum_y_pred', 'accum_y_true_pred', 'accum_metrics'], train_valid_test].any():
                data_2_accum['y_pred'] = model_output.y_pred
                if params.DEBUG:
                    logger.record(f"Epoch Runner registered batch y_pred for accumulation")
                    
            if self.config.loc[['accum_y_true_pred', 'accum_metrics'], train_valid_test].any():
                data_2_accum['y_true'] = batch_device['evaluation_input']['Y']
                if params.DEBUG:
                    logger.record(f"Epoch Runner registered batch y_true for accumulation")

            # Accumulate intermediate layer
            if params.save_intermediate_layer:
                data_2_accum['intermediate_output'] = model_output.intermediate_output
                if params.DEBUG:
                    logger.record(f"Epoch Runner registered batch intermediate_outputs for accumulation")

            # Comuptational report
            if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                self.computation_monitor.checkpoint(name="3_forward")
            
            # ===================== Caluclate loss ============ ===
            # Adjust tensor size
            if self.config.loc[['calc_loss', "calc_metrics"], train_valid_test].any():
                if "W0" in batch_device["evaluation_input"]:
                    for _ in range(len(model_output.y_pred.size())-len(batch_device["evaluation_input"]["W0"].size())):
                        batch_device["evaluation_input"]["W0"] = batch_device["evaluation_input"]["W0"].unsqueeze(-1)
            
            # Calculate loss
            if self.config.loc['calc_loss', train_valid_test]:
                loss = self.calc_loss(params=params, 
                                      train_valid_test=train_valid_test, 
                                      Wpn_idx=Wpn_idx,
                                      batch_device=batch_device, 
                                      model_output=model_output)
                
                # Accumulate loss
                if self.config.loc['accum_loss', train_valid_test]:
                    data_2_accum['loss'] = loss.item()
                    # print(data_2_accum['loss'])
                # Printing
                if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                    print("loss", loss.item())
                    self.computation_monitor.checkpoint(name="4_calc_loss")
            
            # Backward pass (only for training)
            if self.config.loc['with_gradient', train_valid_test]:
                self.scaler.scale(loss + model_output.intra_model_loss * params.INTRA_MODEL_LOSS_FACTOR).backward()
                if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                    self.computation_monitor.checkpoint(name="5_backward")

            # Gradient-driven parms update
            if (i % params.ACCUM_STEP == params.ACCUM_STEP-1 or i == len(pbar)-1) and \
            self.config.loc['model_parm_update', train_valid_test]:
                with torch.autograd.set_detect_anomaly(True):
                    model._update_parameters(self.scaler, epoch, max_norm=params.MAX_NORM)
                    if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                        self.computation_monitor.checkpoint(name="6_model_weight_update")
                    
                    # Save grad snapshot if required
                    save_grad_snapshot = (i==len(pbar)-1 or i==params.ACCUM_STEP-1) and self.config.loc['save_grad_snapshot', train_valid_test]

                    # Evaluate grad
                    self.evaluate_grad(model=model, params=params,
                                       epoch=epoch, train_valid_test=train_valid_test, 
                                       save_grad_snapshot=save_grad_snapshot)
                    if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                        self.computation_monitor.checkpoint(name="7_grad_eval")
                        
                    self.scaler.update() # remember the scaling factors
                    if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                        self.computation_monitor.checkpoint(name="8_scaler_update")
                if i == 0:
                    logger.record("Model parm updated")

            # Accumulate data
            if self.config.loc[[i for i in self.config.index if i.startswith('accum')], train_valid_test].any():
                ## Accum W0 from eval input
                if 'evaluation_input' in batch_device:
                    data_2_accum['weight0'] = batch_device['evaluation_input'].get('W0', 1)
                    if params.DEBUG:
                        print(data_2_accum['weight0'])
                    
                ## gradient for fisher
                if self.config.loc['accum_fisher', train_valid_test]:
                    data_2_accum['model_named_parameters'] = model.named_parameters()

                m_col_names = [k for k in batch_device['model_input'] if k.startswith('M')]
                for col_name in sorted(m_col_names):
                    if col_name in batch_device['model_input']:
                        # print(col_name[col_name.rfind("_")+1:])
                        data_2_accum[f'mask_{col_name[col_name.rfind("_")+1:]}'] = batch_device['model_input'][col_name]

                data_accumulator.add_batch(data_2_accum=data_2_accum, 
                                           params=params, 
                                           epoch_config=self.config[[train_valid_test]], 
                                           logger=logger)

            if computation_monitor and i%self.computation_monitor._freq_batch == 0:
                self.computation_monitor.export_report(epoch=epoch, batch=i, 
                                                       phase=train_valid_test
                                                      ) # This will save already
                
        data_accumulator.save_data(epoch_config=self.config, 
                                   params=params, train_valid_test=train_valid_test)
        
        if params.DEBUG:
            quit()
        epoch_data = data_accumulator.export(params=params, epoch_config=self.config, 
                                             train_valid_test=train_valid_test)
        # print("epoch_run after accumulator export: epoch_data = \n", epoch_data)
        # print("epoch_run: log_dropout = ", self.config.loc['log_dropout', train_valid_test])
        # quit()
        if self.config.loc['accum_metrics', train_valid_test]:
            epoch_data = eval_classification(
                    epoch_data, 
                    return_metrics=self.config.loc['log_metrics', train_valid_test]
                )
        # print("epoch_run after metric eval: epoch_data = \n", epoch_data)
        
        if logbook:
            dropout_report = None
            if self.config.loc['log_dropout', train_valid_test]:
                dropout_report = model.report_dropout()
                # print(dropout_report)
            logbook.write_epoch_data(epoch=epoch, epoch_config=self.config, 
                                     train_valid_test=train_valid_test, 
                                     epoch_data=epoch_data, 
                                     params=params, dropout_report=dropout_report, logger=logger)
            
        if self.config.loc[[idx for idx in self.config.index if idx.startswith("log_")], train_valid_test].any():
            return logbook
    