import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import os
import math
from numpy.core.multiarray import scalar
from modules.data_containers import ModelOutput
from modules.train_control.parameters import Parameters
from modules.file_handling import compress, decompress
from modules.model.part_TIM import TIM
from modules.model.part_SASA import ExposedResiduePrediction
from modules.model.part_HEIWA_Imp import HEIWA_Imp
from modules.model.part_HEIWA_Hybridizer import HEIWA_Hybridizer
from modules.model.part_ThermPred import ThermoPred
from modules.model.basic_layers import *
from modules.train_log import Logger

torch.serialization.add_safe_globals([scalar])

def xavier_initialize(model, module_types: tuple[nn.Module] = (nn.Linear, nn.Embedding, nn.LayerNorm, nn.RMSNorm, 
                                                               Swish, SwiGLU_Simplified)):
    """ nn.Parameter requires torch.randn to initialize """
    for name, m in model.named_modules():
        if m!=model and isinstance(m, module_types):
            if not getattr(m, "pretrained", False):
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        gain = getattr(m, "gain", 1.0) # fall back to the default gain (1.0) if the attr isn't set
                        nn.init.xavier_uniform_(m.weight, gain=gain)
                        if gain != 1.0:
                            print(f"{m}.gain change to {gain}")
                    if hasattr(m, 'bias') and m.bias is not None:
                        bias = getattr(m, "Bias", 0)
                        m.bias.fill_(bias)
                        if bias != 0:
                            print(f"{m}.bias change to {bias}")
                elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if hasattr(m, 'Bias') and m.bias is not None: 
                        m.bias.fill_(0)
                elif isinstance(m, (Swish, SwiGLU_Simplified)):
                    xavier_initialize(model=m)
            
class MyModel(nn.Module):
    def __init__(self, expected_tasks:list=['train', 'valid', 'test'], logger:None=None,
                 status_instructions:pd.Series|None=None, params: Parameters | None=None):
        super().__init__()
        self.logger = Logger(f"{params.DIRECTORIES.MODEL_DIR}/all_tensors.txt")
        self.num_cpu = params.NUM_WORKERS
        self.inference = params.inference
        logger.record(f"inference = {params.inference}")
        self.save_intermediate_layer = getattr(params, "save_intermediate_layer", False)
        logger.record(f"save_intermediate_layer = {self.save_intermediate_layer}")
        self.parts = nn.ModuleDict()

        
        self.parts['TIM'] = TIM(
            in_dim=params.PROT_DIM, out_dim=params.TIM_OUT_DIM, mid_dim=params.TIM_MID_DIM,
            dropout=params.DROPOUT
        )
        self.parts["ExposedResiduePrediction"] = ExposedResiduePrediction(
            dim_prot=params.PROT_DIM, 
            dim_intermediate=params.SASA_INTERMEDIATE_DIM, # sasa_intermediate_dim
            dim_out=1,
            dropout_value=params.DROPOUT["low"]
        )
        
        self.parts['HEIWA_Imp'] = HEIWA_Imp(
            le_cha_bias_action = params.LE_CHA_BIAS_INIT, 
            le_cha_bias_final = params.HEIWA_IMP_LECHA_BIAS, 
            with_scaler=params.HEIWA_IMP_WITH_SCALER, 
            num_solv_head=params.HEIWA_IMP_NUM_SOLVENT_HEAD, 
            num_prot_head=params.HEIWA_IMP_NUM_PROT_HEAD, 
            head_label=params.HEIWA_IMP_HEAD_LABEL, 
            rot_head_idx=params.HEIWA_IMP_ROT_HEAD_IDX, 
            head_require_rot=params.HEIWA_IMP_HEAD_REQUIRE_ROT, 
            num_rot_head=params.NUM_ROT_HEAD, 
            z_dim_to_qk_dim_ratio=params.HEIWA_IMP_Z_QK_DIM_RATIO, 
            dim_qk=params.HEIWA_IMP_QK_DIM,  
            dim_v=params.heiwa_imp_v_dim, # HEIWA_IMP_V_DIM
            dim_prot=params.PROT_DIM, 
            dim_sigmoid_factor=params.HEIWA_IMP_SIGMOID_FACTOR,
            dropout=params.DROPOUT,
        )
        
        self.parts["HEIWA_Hybridizer"] = HEIWA_Hybridizer(
            in_dim=params.heiwa_imp_v_dim*params.HEIWA_IMP_NUM_SOLVENT_HEAD*params.HEIWA_IMP_NUM_PROT_HEAD, 
            v_compress=params.hyb_v_compress, 
            expansion_factor=params.HYB_SWISH_EXPAND, 
            out_dim=params.PROT_DIM, 
            dropout=params.DROPOUT
        )
        
        self.parts['Thermo'] = ThermoPred(in_dim = params.PROT_DIM, 
                                          intermediate_dim=params.THERMO_INTERMEDIATE_DIM,   # thermo_intermediate_dim, thermo_dim[0]
                                          out_dim=params.THERMO_OUT_DIM, 
                                          dropout=params.DROPOUT) # params.DROPOUT["low"]*0.9

        
        # Set requires_grad
        if 'train' in expected_tasks:
            self.grad_schedule = pd.read_csv(f"{params.DIRECTORIES.DATA_PATH}/grad_schedule.csv", 
                                             header=0, index_col='epoch').astype(bool)
            logger.record(f"grad_schedule loaded")
            self.lr_schedule = pd.read_csv(f"{params.DIRECTORIES.DATA_PATH}/lr_schedule.csv", 
                                             header=0, index_col='epoch')
            logger.record(f"lr_schedule loaded")
            for i in self.lr_schedule.index:
                cols = list(self.parts.keys())
                if (self.lr_schedule.loc[i:, cols] == self.lr_schedule.loc[i, cols]).all().all():
                    self.lr_schedule_milestone = i
                    break
            logger.record(f"lr_schedule_milestone set to {self.lr_schedule_milestone}")
            self.lr_decay = params.LR_DECAY
            logger.record(f"lr decay rate set to {self.lr_decay}")
            
            # Initialize optimizers
            self.optimizer = {}
            for part in self.parts:
                if (self.lr_schedule[part]>0).any():
                    logger.record(f"Construct optimizer for {part}")
                    self.optimizer[part] = optim.Adam(
                        self.parts[part].parameters(), 
                        lr=self.lr_schedule.loc[0, part]
                    ) 
                else:
                    for _, parms in self.parts[part].named_parameters():
                        parms.requires_grad=False
            
        if not self.inference:
            if status_instructions is not None:
                if status_instructions["last_epoch"] > 1:
                    self.nondefault_initialize_dropout(status_instructions=status_instructions)

        
            with torch.no_grad():
                logger.record(f"Applying Xavier intiliazation")
                xavier_initialize(model=self)

            # "Explicit", 
            for part in ["TIM", "ExposedResiduePrediction", "HEIWA_Imp", "Thermo"]:
                logger.record(f"Loading Pretrained model weights for {part}")
                self.parts[part]._load_default(
                    logger=logger, 
                    num_cpu=self.num_cpu
                )

        if not os.path.exists(f"{params.DIRECTORIES.LOG_DIR}/model_size_{'_'.join(expected_tasks)}.txt"):
            self._report_model_size(params=params, expected_tasks=expected_tasks)

    def _report_model_size(self, params, expected_tasks):
        with open(f"{params.DIRECTORIES.LOG_DIR}/model_size_{'_'.join(expected_tasks)}.txt", 'a+') as file:
            for part in self.parts:
                file.write(f"{part}\n")
                for k,v in self.parts[part].state_dict().items():
                    file.write(f"{k}\t{v.size()}\t{v.numel()}\n")
                file.write("\n")
        
    def _report_module_names(self, logger):
        logger.record(f"======== Name of modules in MyModel =======\n")
        for part in self.parts:
            for name, module in self.parts[part].named_modules():
                logger.record(f"{name}\n")

    def _set_grad_by_schedule(self, epoch:int, logger):
        print(f"======= Set model gradient schedule =======")
        for part in self.parts:
            logger.record(f"Gradient of {part} set to {self.grad_schedule.loc[epoch, part]}")
            for m in self.parts[part].modules():
                if isinstance(m, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.RMSNorm, nn.Parameter)):
                    m.requires_grad = self.grad_schedule.loc[epoch, part] 

    def _set_lr_by_schedule(self, epoch:int, logger):
        if epoch <= self.lr_schedule_milestone:
            for part in self.optimizer:
                logger.record(f"Setting lr of all param_group in {part} to {float(self.lr_schedule.loc[epoch, part])}")
                for param_group in self.optimizer[part].param_groups:
                    param_group['lr'] = float(self.lr_schedule.loc[epoch, part])
        else:
            for part in self.optimizer:
                logger.record(f"Apply decay of {self.lr_decay} to all param_group in {part}")
                for param_group in self.optimizer[part].param_groups:
                    param_group['lr'] *= self.lr_decay
        
    def nondefault_initialize_dropout(self, status_instructions:pd.Series):
        for part in self.parts:
            for name, module in self.parts[part].named_modules():
                if name in status_instructions:
                    module.p = status_instructions[name]
    
    def _loadpretrain(self, hp_folder: str, epoch: int|str='final', task:str='train', logger=None, device: str = 'cuda'):
        """Load pretrained model weights and optimizer state."""
        def ensure_dropout_tuner_metric(part):
            for m in self.parts[part].modules():
                if isinstance(m, nn.Dropout):
                    m.tuner.metric = m.p
        for part in self.parts:
            # Set the path
            if epoch != 'final':
                model_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_model.pth")
                if task == 'train':
                    optimizer_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_optimizer.pth")
            else:
                # Just find the last epoch
                model_path = os.path.join(hp_folder, f"{part}_{epoch}_model.pth")
                if not os.path.exists(f"{model_path}.gz") and not os.path.exists(f"{model_path}"):
                    import re
                    pattern = rf"{part}_epoch_([0-9]+)_model.pth"
                    matches = [int(re.findall(pattern, file)[0]) for file in os.listdir(hp_folder) if re.findall(pattern, file)]
                    epoch = max(matches)
                    model_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_model.pth")

            decompress(model_path, logger=logger, num_cpu=self.num_cpu)  # No need to write .gz
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.parts[part].load_state_dict(checkpoint)
            compress(model_path, logger=logger, num_cpu=self.num_cpu)

            if epoch != 'final' and task == 'train':
                ensure_dropout_tuner_metric(part)
                logger.record(f"Loaded model checkpoint: {model_path}")

            if task == 'train':
                if self.grad_schedule.loc[epoch, part] and not self.inference:
                    if os.path.exists(f"{optimizer_path}.gz"):
                        logger.record(f"Load optim for {part} from epoch {epoch}")
                        decompress(optimizer_path, logger=logger, num_cpu=self.num_cpu)
                        checkpoint = torch.load(optimizer_path, map_location=device, weights_only=True)
                        self.optimizer[part].load_state_dict(checkpoint)
                        compress(optimizer_path, logger=logger, num_cpu=self.num_cpu)
                        logger.record(f"Loaded optimizer checkpoint: {optimizer_path}")

                
    def _calc_total_grad_norm(self, part):
        total_norm = 0
        for param in self.parts[part].parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _update_parameters(self, scaler, epoch:int, max_norm: float = 0.8, max_tot_grad_norm:float = 1e-6):
        """Perform backpropagation for specific part."""
        for part in self.optimizer:
            if self.grad_schedule.loc[epoch, part]:
                scaler.unscale_(self.optimizer[part])
                torch.nn.utils.clip_grad_norm_(self.parts[part].parameters(), max_norm)
                scaler.step(self.optimizer[part])
        
    def report_weight(self, DF, path, max_sample=5):
        i = len(DF)
        with torch.no_grad():
            for part in self.parts:
                if self.grad_schedule[part].any():
                    for name, parm in self.parts[part].named_parameters():
                        sample = min(max_sample, len(parm))
                        DF.loc[i, [f"{name}_{j}" for j in range(sample)]] = parm.flatten()[:sample].cpu().numpy()
            DF.to_csv(f"{path}", index=False)
    
    def _clear_grad(self, epoch):
        if hasattr(self, 'optimizer'):
            for part in self.optimizer:
                self.optimizer[part].zero_grad()
                self.parts[part].zero_grad()
            
    def update_dropout(self, train_data, valid_data, epoch, logger=None):
        """Update dropout rate for specific part."""
        logger.record("Updating dropout rates...")
        for part in self.parts:
            if self.grad_schedule.loc[epoch-4:epoch, part].any():
                for name, module in self.parts[part].named_modules():
                    if isinstance(module, nn.Dropout):
                        module.tuner(train_metric=train_data, \
                                     valid_metric=valid_data, logger=logger)
                        module.p = module.tuner.metric
                        if logger:
                            logger.record(f"Updated dropout for {name} to {module.p}")
        return self.report_dropout()
            
    def report_dropout(self):
        """Report current dropout rates."""
        report = {}
        for part_name, _ in self.parts.items():
            for name, module in self.parts[part_name].named_modules():
                if isinstance(module, nn.Dropout):
                    report[name] = module.p
        return report
    
    def change_lr(self, part: str, lr: float):
        """Change learning rate for specific part."""
        if part in self.optimizer:
            for param_group in self.optimizer[part].param_groups:
                param_group['lr'] = lr
            print(f"Changed learning rate for {part} to {lr}")
    
    def save_chkpt(self, hp_folder: str, epoch: int, logger=None):
        """Save model weights and optimizer state as checkpoints."""
        os.makedirs(hp_folder, exist_ok=True)
        for part in self.parts:
            # Save model state
            model_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_model.pth")
            torch.save(self.parts[part].state_dict(), model_path)
            compress(model_path, logger=logger, num_cpu=self.num_cpu)
            
            # Save optimizer state
            if not self.inference and part in self.optimizer:
                optimizer_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_optimizer.pth")
                torch.save(self.optimizer[part].state_dict(), optimizer_path)
                compress(optimizer_path, logger=logger, num_cpu=self.num_cpu)

    
    def remove_old_chkpt(self, hp_folder: str,  epoch: int):
        """Remove checkpoints older than 10 epochs."""
        # Unconditionally remove old checkpoints for no_optimizer parts
        for part in self.parts:
            if not part in self.optimizer:
                for ep in range(epoch-1):
                    old_model_path = os.path.join(hp_folder, f"{part}_epoch_{ep}_model.pth.gz")
                    if os.path.exists(old_model_path):
                        os.remove(old_model_path)
        # Remove very old chpt if the there is optimizer
        if epoch > 5:
            for part in self.optimizer:
                for ep in range(epoch-15):
                    if ep %10 != 5 or ep < 2:
                        old_model_path = os.path.join(hp_folder, f"{part}_epoch_{ep}_model.pth.gz")
                        old_optimizer_path = os.path.join(hp_folder, f"{part}_epoch_{ep}_optimizer.pth.gz")
                        
                        if os.path.exists(old_model_path):
                            os.remove(old_model_path)
                        if os.path.exists(old_optimizer_path):
                            os.remove(old_optimizer_path)
                    
                            
    def decide_final(self, hp_folder: str, epoch: int, hp_tuning:bool):
        """Decide which checkpoint should be saved as the final model."""
        """ If hp_tuning, do not name the best model final yet """
        for part in self.parts:
            best_model_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_model.pth.gz")
            best_optimizer_path = os.path.join(hp_folder, f"{part}_epoch_{epoch}_optimizer.pth.gz")

            if os.path.exists(best_model_path) and not hp_tuning:
                final_model_path = os.path.join(hp_folder, f"{part}_final_model.pth.gz")
                final_optimizer_path = os.path.join(hp_folder, f"{part}_final_optimizer.pth.gz")
                
                # Copy best model as final model
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                
                if os.path.exists(best_optimizer_path):
                    shutil.copy2(best_optimizer_path, final_optimizer_path)
        print(f"Saved final model from epoch {epoch}")

    def update_le_cha_bias(self, params, task, epoch, logger):
        if task == 'train':
            for part in self.parts: # 
                if part in ("HEIWA_Imp", "HEIWA_Exp"):
                    self.parts[part].le_cha_bias_action = params.LE_CHA_BIAS_INIT
                    for ep in range(epoch+1):
                        self.parts[part].le_cha_bias_action += params.LE_CHA_BIAS_STEP * (1+0.5*ep)
                    logger.record(f"Minimum Le Chatelier's bias default of {part} set to {self.parts[part].le_cha_bias_action}")
                    logger.record(f"Actual Le Chatelier's bias of {part} set to {torch.where(self.parts[part].le_cha_bias_final > self.parts[part].le_cha_bias_action, self.parts[part].le_cha_bias_final, self.parts[part].le_cha_bias_action)}")
        else:
            for part in self.parts:
                if part in ("HEIWA_Imp", "HEIWA_Exp"):
                    self.parts[part].le_cha_bias_action = -999
                    logger.record(f"Minimum Le Chatelier's bias default of {part} set to {self.parts[part].le_cha_bias_action}")
                    logger.record(f"Actual Le Chatelier's bias of {part} set to {torch.where(self.parts[part].le_cha_bias_final > self.parts[part].le_cha_bias_action, self.parts[part].le_cha_bias_final, self.parts[part].le_cha_bias_action)}")

    def log_tensor(self, tensor, tensor_name):
        self.logger.record(f"{tensor_name}")
        self.logger.record(f"Max: {tensor.max().item()}")
        self.logger.record(f"Mean: {tensor.mean().item()}")
        self.logger.record(f"Min: {tensor.min().item()}")
        if tensor.numel() > 1:
            self.logger.record(f"Std: {tensor.std().item()}")
        self.logger.record(f"Count>0.5: {(tensor>0.5).sum().item()}")
        self.logger.record(f"Count>0: {(tensor>0).sum().item()}")
        self.logger.record(f"Count=0: {(tensor==0).sum().item()}")
        self.logger.record(f"Count<0: {(tensor<0).sum().item()}")
        
        tensor_clone = tensor.clone()
        tensor_clone = tensor_clone.reshape(-1)
        if len(tensor_clone) == 1:
            self.logger.record(f"{tensor_clone.item()}\n") # show the 1st sample in the batch
        else:
            self.logger.record(f"{tensor_clone[:min(10, len(tensor_clone))]}\n")
        
    def forward(self, model_input: Dict[str, torch.Tensor], log_all_tensors=False) -> ModelOutput:
        """Forward pass through the model."""
        bs, l, _ = model_input['X'].size()

        if log_all_tensors:
            self.logger.record(f"******* Start recording tensors ***********")
            
        tim_output = self.parts['TIM'](model_input["X"])
        tim_gate = tim_output.y_pred
            
        ExposedResiduePrediction_output = self.parts["ExposedResiduePrediction"](model_input["X"])
        sasa_gate = ExposedResiduePrediction_output.y_pred

        if log_all_tensors:
            self.log_tensor(tim_gate, "tim_gate")
            self.log_tensor(sasa_gate, "sasa_gate")
            
        # Implicit
        heiwa_imp_output = self.parts['HEIWA_Imp'](x=model_input["X"])
        water_modification_imp = heiwa_imp_output.y_pred
        imp_loss = heiwa_imp_output.intra_model_loss
        na_atten_imp, cl_atten_imp = heiwa_imp_output.intermediate_output
        if log_all_tensors:
            self.log_tensor(water_modification_imp, "water_modification_imp")
            self.log_tensor(na_atten_imp, "h_atten_imp", )
            self.log_tensor(cl_atten_imp, "l_atten_imp", )
            
        # Compile intramodel loss
        intra_loss = imp_loss / (bs**0.5 * 100**0.5)

        # Hybydridizer
        hyd_output = self.parts["HEIWA_Hybridizer"](
            x_imp = water_modification_imp, 
            tim_gate = tim_gate, 
            sasa_gate = sasa_gate
        )
        water_modification = hyd_output.y_pred
        
        # Thermo (There is mask already)
        thermo_output = self.parts['Thermo'](
            model_input=model_input, 
            water_modification=water_modification, 
            save_intermediate_layer=self.save_intermediate_layer
        ) # Masking applied inside

        if log_all_tensors:
            self.log_tensor(water_modification, "water_modification")
            self.log_tensor(imp_loss, "imp_loss")
            self.log_tensor(water_modification.abs().sum(), "water_modification_regularization")
            self.log_tensor(thermo_output.y_pred, "thermo_output.y_pred")
            
        return ModelOutput(
            y_pred=thermo_output.y_pred,  # 
            intermediate_output={"final_layer": thermo_output.intermediate_output}, # thermo_output.intermediate_output
            intra_model_loss=intra_loss
        ) # 
        
