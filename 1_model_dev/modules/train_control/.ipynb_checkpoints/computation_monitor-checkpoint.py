import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
from modules.train_control.parameters import Parameters

@dataclass
class CheckpointReport:
    """ Container for a single checkpoint's measurements """
    name: str
    time_elapsed: float  # seconds since last checkpoint
    cumulative_time: float  # seconds since start
    memory_allocated: float  # GB
    memory_reserved: float  # GB


@dataclass
class ComputationReport:
    """Container for computational statistics"""
    checkpoints: List[CheckpointReport] = field(default_factory=list)
    peak_memory: float = 0.0  # GB
    total_time: float = 0.0  # seconds
    profiler_stats: Optional[str] = None
    flops_by_module: Dict[str, int] = field(default_factory=dict)
    
    def save(self, save_dir: Path, epoch: int, batch: int, phase: str):
        """
        Save computation report to a single CSV file.
        
        Args:
            save_dir: Directory to save report
            epoch: Current epoch number
            phase: Phase name ('train', 'valid', 'test', 'inference')
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the row data
        row_data = {
            'epoch': epoch,
            'phase': phase,
            'total_time_s': self.total_time,
            'peak_memory_gb': self.peak_memory,
        }
        
        # Add checkpoint timings and memory
        for cp in self.checkpoints:
            row_data[f'{cp.name}_time_s'] = cp.time_elapsed
            row_data[f'{cp.name}_memory_gb'] = cp.memory_allocated
            row_data[f'{cp.name}_cumulative_time_s'] = cp.cumulative_time
        
        # Add total FLOPs
        if self.flops_by_module:
            total_flops = sum(self.flops_by_module.values())
            row_data['total_gflops'] = total_flops / 1e9
            
            # Add top modules (e.g., top 5 by FLOPs)
            sorted_modules = sorted(
                self.flops_by_module.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for i, (module_name, flops) in enumerate(sorted_modules[:5]):
                row_data[f'top{i+1}_module'] = module_name
                row_data[f'top{i+1}_gflops'] = flops / 1e9
        
        # Save to CSV
        csv_file = save_dir / f"computation_report_ep{epoch}_{phase}_btch{batch}.csv"
        df = pd.DataFrame([row_data])
        
        if csv_file.exists():
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)
        
        # Optionally save profiler stats to separate text file
        if self.profiler_stats:
            profiler_file = save_dir / f"profiler_ep{epoch}_{phase}_btch{batch}.txt"
            with open(profiler_file, 'w') as f:
                f.write(self.profiler_stats)

class FlopCounter:
    """ Count FLOPs using forward hooks """
    
    def __init__(self):
        self.flop_counts = {}
        self.hooks = []
    
    def register_hooks(self, model: torch.nn.Module):
        """ Register hooks on all leaf modules """
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(self._make_hook(name, module))
                self.hooks.append(hook)
    
    def _make_hook(self, name: str, module: torch.nn.Module):
        """ Create a hook function for a specific module """
        def hook(module, input, output):
            flops = self._compute_flops(module, input, output)
            self.flop_counts[name] = self.flop_counts.get(name, 0) + flops
        return hook
    
    def _compute_flops(self, module, input, output) -> int:
        """ Compute FLOPs for common layer types """
        flops = 0
        
        if isinstance(module, torch.nn.Linear):
            batch_size = input[0].numel() // input[0].shape[-1]
            flops = batch_size * module.in_features * module.out_features * 2
            
        elif isinstance(module, torch.nn.Conv2d):
            batch_size = output.shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            flops = (batch_size * out_h * out_w * 
                    module.in_channels * kernel_ops * 
                    module.out_channels * 2)
            
        elif isinstance(module, torch.nn.MultiheadAttention) or 'Attention' in module.__class__.__name__:
            inp = input[0] if isinstance(input, tuple) else input
            seq_len = inp.shape[1] if inp.dim() >= 2 else inp.shape[0]
            d_model = inp.shape[-1]
            flops = 2 * seq_len * seq_len * d_model
        
        return flops
    
    def get_total_flops(self) -> Dict[str, int]:
        """ Get FLOP counts by module """
        return self.flop_counts.copy()
    
    def reset(self):
        """ Reset counters """
        self.flop_counts.clear()
    
    def remove_hooks(self):
        """ Remove all registered hooks """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ComputationMonitor:
    """ Monitor computational costs during training/inference.
    
    Usage:
        from parameters import Parameters
        params = Parameters()
        monitor = ComputationMonitor(params, enable_profiler=False, enable_flops=True)
        monitor.register_model(model)
        
        for epoch in range(num_epochs):
            # Training phase
            monitor.start()
            output = model(input)
            monitor.checkpoint("forward")
            loss = criterion(output, target)
            monitor.checkpoint("loss")
            loss.backward()
            monitor.checkpoint("backward")
            optimizer.step()
            monitor.checkpoint("optimizer_step")
            
            report = monitor.export_report()
            report.save(monitor.save_dir, epoch=epoch, phase='train')
            
            # Validation phase
            monitor.start()
            with torch.no_grad():
                output = model(val_input)
                monitor.checkpoint("forward")
                val_loss = criterion(output, val_target)
                monitor.checkpoint("loss")
            
            report = monitor.export_report()
            report.save(monitor.save_dir, epoch=epoch, phase='valid')
    """
    
    def __init__(self, 
                 params: Parameters,  # Your Parameters object
                 enable_profiler: bool = True,
                 enable_flops: bool = True,
                 device: str = 'cuda'):
        """ Args:
            params: Parameters object containing SAVE_DIR attribute
            enable_profiler: Use torch.profiler (more overhead but detailed)
            enable_flops: Count FLOPs using hooks (some overhead)
            device: 'cuda' or 'cpu'
        """
        self.save_dir = Path(params.DIRECTORIES.MODEL_DIR) / "computation_reports"
        self.enable_profiler = enable_profiler
        self.enable_flops = enable_flops
        self.device = device

        self._freq_batch = params.COMPUTE_MONITOR_FREQ_BATCH
        self._start_time = None
        self._last_checkpoint_time = None
        self._profiler = None
        self._flop_counter = None
        self._checkpoints = []
        self._peak_memory = 0.0
        
    def start(self):
        """ Start monitoring at beginning of iteration """
        # Reset state
        self._checkpoints = []
        self._peak_memory = 0.0
        self._start_time = time.time()
        self._last_checkpoint_time = self._start_time
        
        # Reset GPU memory tracking
        # logger.record(f"Verify if device is cuda, if cuda is available?")
        if self.device == 'cuda' and torch.cuda.is_available():
            # logger.record(f"Yes! reset_peak_memory_stats + synchronize")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Start profiler if enabled
        
        if self.enable_profiler:
            # logger.record(f"Profiler enabled")
            from torch.profiler import profile, ProfilerActivity
            activities = [ProfilerActivity.CPU]
            if self.device == 'cuda':
                activities.append(ProfilerActivity.CUDA)
            
            self._profiler = profile(
                activities=activities,
                record_shapes=True,
                with_stack=False,
            )
            self._profiler.start()
        
        # Reset FLOP counter if enabled
        if self.enable_flops and self._flop_counter is not None:
            # logger.record(f"FLOP counter reset")
            self._flop_counter.reset()
    
    def checkpoint(self, name: str):
        """ Record computational state at a checkpoint """
        # Synchronize GPU for accurate timing
        # logger.record(f"Verify if device is cuda, if cuda is available?")
        if self.device == 'cuda' and torch.cuda.is_available():
            # logger.record(f"Yes! Synchronize")
            torch.cuda.synchronize()

        current_time = time.time()
        time_elapsed = current_time - self._last_checkpoint_time
        cumulative_time = current_time - self._start_time
        
        # Get memory stats
        if self.device == 'cuda' and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            self._peak_memory = max(self._peak_memory, peak_memory)
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
        
        # Create checkpoint record
        cp = CheckpointReport(
            name=name,
            time_elapsed=time_elapsed,
            cumulative_time=cumulative_time,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
        )
        
        self._checkpoints.append(cp)
        self._last_checkpoint_time = current_time
    
    def export_report(self, epoch:int, batch:int, phase:str):
        """ Finalize monitoring and return report """
        # Final synchronization
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - self._start_time
        
        # Stop profiler and get stats
        profiler_stats = None
        if self.enable_profiler and self._profiler is not None:
            self._profiler.stop()
            sort_key = "cuda_time_total" if self.device == 'cuda' else "cpu_time_total"
            profiler_stats = self._profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=20
            )
        
        # Get FLOP counts
        flops_by_module = {}
        if self.enable_flops and self._flop_counter is not None:
            flops_by_module = self._flop_counter.get_total_flops()
        
        report =  ComputationReport(
            checkpoints=self._checkpoints.copy(),
            peak_memory=self._peak_memory,
            total_time=total_time,
            profiler_stats=profiler_stats,
            flops_by_module=flops_by_module,
        )
        report.save(save_dir=self.save_dir, epoch=epoch, batch=batch, phase=phase)
    
    def register_model(self, logger, model: torch.nn.Module):
        """Register model for FLOP counting (call once after model creation)"""
        if self.enable_flops:
            if self._flop_counter is None:
                self._flop_counter = FlopCounter()
            logger.record(f"Count FLOP")
            self._flop_counter.register_hooks(model)

