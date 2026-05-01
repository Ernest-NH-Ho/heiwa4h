# modules/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional
import os
import glob
from .model_microlayer import SwiGLU_FFN_xResNet
from .train_earlystop import EarlyStopping_max
from .train_hp_control import Metric_tuner_double_bounded
from .data_containers import ModelOutput
from .parameters import Parameters

def xavier_initialize(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding)):
    with torch.no_grad():
        for m in self.modules():
            if isinstance(m, module_types):
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
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None: 
                    m.bias.fill_(0)
                    
class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(FFN, self).__init__()
        # self.embed = nn.Embedding(21, 16)
        self.ffn11 = nn.Linear(in_dim, out_dim)
        self.dropout11 = nn.Dropout(dropout)

    def forward(self, model_input: Dict[str, torch.Tensor]) -> ModelOutput:
        """Forward pass through the model."""
        # if torch.isnan(model_input['x']).any():
        #     print("NaN in input")
        a = self.ffn11(model_input['x'])
        # if torch.isnan(a).any():
        #     print("NaN in a b4 dropout")
        a = self.dropout11(a)
        a = torch.clamp(a, min=-8, max=8)
        return F.sigmoid(a)
        
class MyModel(nn.Module):
    def __init__(self, early_stopping: bool|None = None, params: Parameters | None=None):
        super(MyModel, self).__init__()
        
        self.inference = params.inference
        self.component = nn.ModuleDict()
        self.component['FFN'] = FFN(in_dim=params.IN_DIM*3, out_dim=params.OUT_DIM, 
                                    dropout=params.DROPOUT)
        
        if not self.inference:
            # Initialize optimizers
            self.optimizer = {}
            self.optimizer['FFN'] = optim.Adam(
                self.component['FFN'].parameters(), 
                lr=params.LEARNING_RATE
            )
            
            # Initialize dropout tuners
            self.dropout_tuner = {}
            self.dropout_tuner['FFN'] = Metric_tuner_double_bounded(
                metric=params.DROPOUT, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.015, thres_Q1=1.009, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
            
            # Initialize early stopping
            self.early_stop = {}
            if early_stopping==None:
                self.early_stop['FFN'] = EarlyStopping_max(
                    verbose=True, 
                    save=True,
                    data_target='FFN', 
                    params = params
                )
        xavier_initialize(self)
        
    def _loadpretrain(self, hp_folder: str, component: str, epoch: int|str='final', device: str = 'cuda'):
        """Load pretrained model weights and optimizer state."""
        if epoch != 'final':
            model_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_model.pth")
            optimizer_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_optimizer.pth")
        else:
            model_path = os.path.join(hp_folder, f"{component}_{epoch}_model.pth")
            optimizer_path = os.path.join(hp_folder, f"{component}_{epoch}_optimizer.pth")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.component[component].load_state_dict(checkpoint)
            print(f"Loaded model checkpoint: {model_path}")
        
        if os.path.exists(optimizer_path) and not self.inference:
            checkpoint = torch.load(optimizer_path, map_location=device, weights_only=True)
            self.optimizer[component].load_state_dict(checkpoint)
            print(f"Loaded optimizer checkpoint: {optimizer_path}")

    def _calc_total_grad_norm(self, compo):
        total_norm = 0
        for param in self.component[compo].parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _optimizer_backpropagation(self, scaler, component: str, max_norm: float = 0.8, max_tot_grad_norm:float = 1e-6):
        """Perform backpropagation for specific component."""
        # if component in self.optimizer:
        # total_norm = self._calc_total_grad_norm(component)
        # if total_norm < max_tot_grad_norm:
        #     print(total_norm)
        #     dropout = self.component[component].dropout.p
        #     self.update_dropout(0, component)
        scaler.unscale_(self.optimizer[component])
        torch.nn.utils.clip_grad_norm_(self.component[component].parameters(), max_norm)
        scaler.step(self.optimizer[component])
        # if total_norm < max_tot_grad_norm:
        #     self.update_dropout(dropout, component)

    def report_weight(self, DF, path, max_sample=5):
        i = len(DF)
        with torch.no_grad():
            for name, parm in self.named_parameters():
                sample = min(max_sample, len(parm))
                DF.loc[i, [f"{name}_{j}" for j in range(sample)]] = parm.flatten()[:sample].cpu().numpy()
            DF.to_csv(f"{path}", index=False)
    
    def _clear_grad(self, compo):
        if hasattr(self, 'optimizer'):
            self.optimizer[compo].zero_grad()
        self.component[compo].zero_grad()
            
    def update_dropout(self, dropout: float, component: str, logger=None):
        """Update dropout rate for specific component."""
        for module in self.component[component].modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        if logger != None:
            logger.record(f"Updated dropout for {component} to {dropout}")
            
    def report_dropout(self, logger):
        """Report current dropout rates."""
        for comp_name, _ in self.component.items():
            for module in self.component[comp_name].modules():
                if isinstance(module, nn.Dropout):
                    logger.record(f"Component {comp_name} dropout rate: {module.p}")
                    continue
    
    def change_lr(self, component: str, lr: float):
        """Change learning rate for specific component."""
        if component in self.optimizer:
            for param_group in self.optimizer[component].param_groups:
                param_group['lr'] = lr
            print(f"Changed learning rate for {component} to {lr}")
    
    def save_component(self, hp_folder: str, component: str, epoch: int):
        """Save model weights and optimizer state as checkpoints."""
        os.makedirs(hp_folder, exist_ok=True)
        if component == 'all':
            for compo in self.component:
                self.save_component(hp_folder, compo, epoch)
        else:
            # Save model state
            model_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_model.pth")
            torch.save(self.component[component].state_dict(), model_path)
            
            # Save optimizer state
            if not self.inference and component in self.optimizer:
                optimizer_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_optimizer.pth")
                torch.save(self.optimizer[component].state_dict(), optimizer_path)
    
    def remove_old_chkpt(self, hp_folder: str, component: str, epoch: int):
        """Remove checkpoints older than 10 epochs."""
        if epoch > 10:
            for ep in range(1, epoch-10):
                if ep %10 != 0:
                    old_model_path = os.path.join(hp_folder, f"{component}_epoch_{ep}_model.pth")
                    old_optimizer_path = os.path.join(hp_folder, f"{component}_epoch_{ep}_optimizer.pth")
                    
                    if os.path.exists(old_model_path):
                        os.remove(old_model_path)
                    if os.path.exists(old_optimizer_path):
                        os.remove(old_optimizer_path)
        
    def decide_final(self, hp_folder: str, component: str, epoch: int):
        """Decide which checkpoint should be saved as the final model."""
        best_model_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_model.pth")
        final_model_path = os.path.join(hp_folder, f"{component}_final_model.pth")
        
        best_optimizer_path = os.path.join(hp_folder, f"{component}_epoch_{epoch}_optimizer.pth")
        final_optimizer_path = os.path.join(hp_folder, f"{component}_final_optimizer.pth")
        
        if os.path.exists(best_model_path):
            # Copy best model as final model
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            
            if os.path.exists(best_optimizer_path):
                shutil.copy2(best_optimizer_path, final_optimizer_path)
            
            print(f"Saved final model from epoch {epoch}")
    
    def allEarlyStop(self) -> bool:
        """Check if all components are ready to stop training."""
        return all([self.early_stop[compo].ready_to_stop for compo in self.early_stop])
    
    def forward(self, model_input: Dict[str, torch.Tensor]) -> ModelOutput:
        """Forward pass through the model."""
        
        y_pred = self.component['FFN'](model_input)
        # print(y_pred)
        return ModelOutput(y_pred=y_pred)
        