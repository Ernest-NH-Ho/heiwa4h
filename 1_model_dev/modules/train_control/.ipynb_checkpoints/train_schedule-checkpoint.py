import pandas as pd
from typing import Dict, Any

class TrainSchedule:
    """Class to manage training schedule with pandas dataframe."""
    
    def __init__(self, max_epochs: int = 120, batch_size: int = 128):
        """   Initialize training schedule.
        
        Args:
            max_epochs: Maximum number of epochs
            batch_size: Default batch size        """
        self.max_epochs = max_epochs
        
        # Create schedule dataframe
        self.df = self._create_schedule(max_epochs, batch_size)
    
    def _create_schedule(self, max_epochs: int, batch_size: int) -> pd.DataFrame:
        """ Create the training schedule dataframe. """
        # Create the pattern [False, False, False, True] * 30
        pattern_size = 4
        learn_curve_pattern = [False, False, False, True] * (max_epochs // pattern_size + 1)
        dropout_tuner_pattern = [False] + [False, False, False, True] * (max_epochs // pattern_size + 1)
        early_stop_pattern = [True] + [False, False, False, True] * (max_epochs // pattern_size + 1)
        computation_monitor = [True, False, False, False, False] * (max_epochs // pattern_size + 1)
        
        # Trim to exact size needed
        learn_curve_pattern = learn_curve_pattern[:max_epochs]
        dropout_tuner_pattern = dropout_tuner_pattern[:max_epochs]
        early_stop_pattern = early_stop_pattern[:max_epochs]
        
        # Create dataframe
        schedule_data = {
            'learn_curve': pd.Series(learn_curve_pattern),
            'dropout_tuner': pd.Series(dropout_tuner_pattern),
            'batch_size': pd.Series([batch_size] * max_epochs), 
            'early_stop': pd.Series(early_stop_pattern), 
            'computation_monitor': pd.Series(computation_monitor)
        }
        
        return pd.DataFrame(schedule_data)
    
    def instruct(self, epoch: int) -> Dict[str, Any]:
        """ Get instructions for a specific epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Dictionary with instructions for the epoch          """
        if epoch >= len(self.df):
            # Return default instructions if epoch exceeds schedule
            return self.df.loc[epoch].to_dict()
        
        row = self.df.iloc[epoch]
        return row.to_dict()
    
    def update_batch_size(self, epoch: int, new_batch_size: int):
        """Update batch size for a specific epoch."""
        if epoch < len(self.df):
            self.df.iloc[epoch, self.df.columns.get_loc('batch_size')] = new_batch_size
    
    def get_schedule_summary(self) -> str:
        """Get a summary of the training schedule."""
        summary = f"Training Schedule Summary (Total Epochs: {len(self.df)}):\n"
        summary += f"Learn Curve plotting epochs: {sum(self.df['learn_curve'])}\n"
        summary += f"Dropout tuning epochs: {sum(self.df['dropout_tuner'])}\n"
        summary += f"Default batch size: {self.df['batch_size'].iloc[0]}\n"
        return summary
        