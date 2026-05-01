import os
import pickle
import matplotlib.pyplot as plt
from typing import List, Any
import numpy as np
from modules.train_control.parameters import Parameters
from modules.data_containers import EpochData
from modules.plot import plot_LearnCurve

class Logger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        
    def record(self, content: str, verbose: bool = True):
        if verbose:
            print(content)
        with open(f"{self.out_dir}", "a+") as out:
            out.write(f"{content}\n")

import pandas as pd
import matplotlib.pyplot as plt
import datetime

class LogBook:
    """Class to record and visualize training progress using pandas DataFrame."""
    
    def __init__(self, params):
        # Define all columns for the DataFrame
        self.columns = ['epoch', 'train_valid_test', 'loss'] + params.metric_keys
        self.valid_phases = ['train', 'valid', 'test', 'inference']
        
        # Initialize empty DataFrame
        self.df = pd.DataFrame(columns=self.columns)

    def get_best_epoch(self):
        return self.df.loc[self.df["train_valid_test"]=="test", "epoch"].max()
        
    def extract_metric_1phase(self, params, phase):
        """ sort according to epoch, get the metrics for one phase --> np.array """
        df = self.df.sort_values(by=["epoch"], axis=0).reset_index(drop=True)
        return df.loc[df["train_valid_test"]==phase, params.EARLYSTOP_METRICS].values.copy()
        
    def write_epoch_data(self, epoch: int, epoch_config:pd.DataFrame, train_valid_test: str, 
                         epoch_data: EpochData, dropout_report: dict, params: Parameters|None=None, logger=None):
        """Record epoch data for the specified phase by adding a new row to DataFrame."""
        if logger:
            logger.record(f"Recording {train_valid_test} data for epoch {epoch}")
        if len(self.df[(self.df["train_valid_test"]==train_valid_test) & (self.df["epoch"]==epoch)]) == 0:
            i = len(self.df)
        else:
            i = self.df[(self.df["train_valid_test"]==train_valid_test) & (self.df["epoch"]==epoch)].index[0]
        if i > 5 and train_valid_test=='valid':
            print(i)
        self.df.loc[i, 'train_valid_test'] = train_valid_test
        self.df.loc[i, 'epoch'] = epoch
        if hasattr(epoch_data, 'loss'):
            self.df.loc[i, 'loss'] = epoch_data.loss
        if dropout_report is not None:
            for k,v in dropout_report.items():
                self.df.loc[i, k] = v
        if epoch_config.loc['log_metrics', train_valid_test]:
            for metric in params.metric_keys:
                self.df.loc[i, metric] = epoch_data.metrics[metric]
            
        if logger:
            logger.record(f"Added {train_valid_test} data for epoch {epoch} to LogBook")
    
    def plot_LearnCurve(self, title: str = None, save_dir: str = None):
        """Plot learning curves for all metrics using DataFrame data."""
        
        if self.df.empty:
            print("No data to plot.")
            return

        for data_type in [["acc", "f1"], ["sens", "spec"], "prec", "loss", "dropout"]:
            plot_LearnCurve(
                data_type=data_type, 
                data=self.df, 
                out_folder=save_dir
            )

    def save_csv(self, filename: str = None, save_dir: str = "./"):
        """Save LogBook DataFrame as CSV file."""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logbook_{timestamp}.csv"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Save DataFrame to CSV
        self.df.to_csv(f"{save_dir}/{filename}", index=False)
        print(f"LogBook saved to CSV: {save_dir}/{filename}")
        # return str(file_path)
            
   
    def load_csv(self, file_path: str, merge: bool = True):
        """Load LogBook data from CSV file."""
        
        # Load DataFrame from CSV
        loaded_df = pd.read_csv(file_path, header=0)
        
        # Validate required columns
        required_cols = ['epoch', 'train_valid_test', 'loss']
        missing_cols = [col for col in required_cols if col not in loaded_df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        if merge and not self.df.empty:
            # Merge with existing data
            # Remove duplicates by keeping the loaded data (more recent)
            combined_df = pd.concat([self.df, loaded_df], ignore_index=True)
            
            # Remove duplicates based on epoch and train_valid_test, keeping last occurrence
            combined_df = combined_df.drop_duplicates(
                subset=['epoch', 'train_valid_test'], 
                keep='last'
            ).reset_index(drop=True)
            
            self.df = combined_df
            print(f"LogBook data merged from CSV: {file_path}")
            print(f"Total rows after merge: {len(self.df)}")
        else:
            # Replace existing data
            self.df = loaded_df
            print(f"LogBook data loaded from CSV: {file_path}")
            print(f"Loaded {len(self.df)} rows")
        
        # Sort by epoch and train_valid_test for consistency
        # self.df = self.df.sort_values(['epoch', 'train_valid_test']).reset_index(drop=True)
        # return True
            
    def get_summary(self, params):
        """Get a summary of recorded data from DataFrame."""
        if self.df.empty:
            return {"message": "No data recorded"}
        
        summary = {}
        
        # Overall statistics
        summary['total_records'] = len(self.df)
        summary['phases'] = sorted(self.df['train_valid_test'].unique().tolist())
        
        # Epoch information
        epoch_data = self.df[self.df['train_valid_test'].isin(['train', 'valid'])]
        if not epoch_data.empty:
            summary['epochs_recorded'] = sorted(epoch_data['epoch'].unique().tolist())
            summary['latest_epoch'] = epoch_data['epoch'].max()
            summary['total_epochs'] = len(epoch_data['epoch'].unique())
        
        # Latest metrics by phase
        for phase in summary['phases']:
            phase_data = self.df[self.df['train_valid_test'] == phase]
            if not phase_data.empty:
                latest_record = phase_data.iloc[-1]
                summary[f'latest_{phase}_metrics'] = {
                    col: latest_record[col] for col in params.metric_keys + ['loss', 'dropout']
                    if col in latest_record
                }
        
        # Best performance (for validation data)
        valid_data = self.df[self.df['train_valid_test'] == 'valid']
        if not valid_data.empty:
            best_performance = {}
            
            # Best metrics (higher is better)
            for metric in ['acc', 'f1', 'sens', 'spec', 'prec']:
                if metric in valid_data.columns and not valid_data[metric].isna().all():
                    best_idx = valid_data[metric].idxmax()
                    best_row = valid_data.loc[best_idx]
                    best_performance[f'best_{metric}'] = {
                        'value': best_row[metric],
                        'epoch': best_row['epoch']
                    }
            
            # Best loss (lower is better)
            if 'loss' in valid_data.columns and not valid_data['loss'].isna().all():
                best_idx = valid_data['loss'].idxmin()
                best_row = valid_data.loc[best_idx]
                best_performance['best_loss'] = {
                    'value': best_row['loss'],
                    'epoch': best_row['epoch']
                }
            
            summary['best_performance'] = best_performance
        
        return summary
    
    def get_phase_data(self, epoch, phase: str):
        """Get data for a specific phase (train/valid/test/inference)."""
        # if phase not in ['train', 'valid', 'test', 'inference']:
        #     raise ValueError(f"Phase must be one of ['train', 'valid', 'test', 'inference'], got '{phase}'")
        # print(self.df)
        # quit()
        tmp = self.df[self.df['train_valid_test'] == phase].sort_values(by=['epoch'], ascending=True).reset_index(drop=True).copy()
        # print(len(tmp)-4)
        return tmp.loc[max(0, epoch+1-4):epoch].reset_index(drop=True).copy()

            
