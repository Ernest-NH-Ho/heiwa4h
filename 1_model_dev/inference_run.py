import argparse
import os
import inspect
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
import traceback
import pandas as pd
from modules.train_control.parameters import ConstantParameters, TunableParameters
from modules.epoch_control.epoch_run import EpochRunner
from modules.train_control.hp_tuning import HyperparameterTuner, HyperparameterTuner_Instruction # , HyperparameterTuning_Report
from modules.train_control.utils import import_data
from modules.train_control.components import ConstantComponents, TunableComponents
from modules.dataloader import compile_data_single

def inference_run(components:ConstantComponents, params:TunableParameters, test_epoch=None, hp_instructions:HyperparameterTuner_Instruction|None=None):
    """Main training function."""
    try:
        # Initialize components
        assert hasattr(components, "tasks")
        components = TunableComponents(
            params=params, test_epoch="final" if not hasattr(params, "infer_epoch") else params.infer_epoch, 
            hp_instructions=hp_instructions, components=components
        )
        assert hasattr(components, "tasks")
        
        components.logger_tunable.record("="*50)
        components.logger_tunable.record("INFERENCE SESSION STARTED")
        components.logger_tunable.record("="*50)
        
        test_epoch = components.logbook.get_best_epoch()

        # Random seed
        random_seeds = components.rs_generator.generate(test_epoch)
        torch.manual_seed(random_seeds)
        components.logger_tunable.record(f"Random seed set to {random_seeds}")
        
        # Log parameters
        components.logger_tunable.record(f"Device: {params.DEVICE}")
        components.logger_tunable.record(f"Batch size: {params.BATCH_SIZE}")
        
        # Initialize epoch runner
        epoch_runner = EpochRunner(params=params, expected_tasks=['inference'])
        updated_dropout = None
        
        # Load the best model
        components.model._loadpretrain(
            hp_folder=params.DIRECTORIES.MODEL_DIR, epoch=test_epoch,
            task='inference', device=params.DEVICE, 
            logger=components.logger_tunable
        )
        if params.TUNE_LE_CHA_BIAS:
            components.model.update_le_cha_bias(params=params, epoch="final", task="inference", logger=components.logger_tunable)
        
        # Run test
        inference_loader = compile_data_single(
            target='inference', 
            dataset=components.dataset_inference,
            params=params
        )
        
        components.logger_tunable.record(f"Data loaded: Inference={len(inference_loader.dataset)}")
        
        epoch_runner.run(
            train_valid_test='inference',
            data_source=inference_loader,
            model=components.model,
            epoch=test_epoch,
            logger=components.logger_tunable,
            params=params
        )
        
        
        components.logger_tunable.record(f"\n Inference session completed successfully!")
        components.logger_tunable.record("="*50)
        
    except Exception as e:
        with open(f"{params.DIRECTORIES.LOG_DIR}/error.txt", "a+") as out:
            out.write(f"Combination: \n{hp_instructions}\n\n")
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n")
            quit()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that takes an input value.")
    parser.add_argument("-save_intermediate_layer", type=int, required=True, help="Output interdiate layer for post-training analysis?")
    parser.add_argument("-infer_epoch", type=int, required=False, help="Which model to load?")
    parser.add_argument("-head", type=int, required=True, help="How many implicit solvent head")
    args = parser.parse_args()
    save_intermediate_layer = bool(args.save_intermediate_layer)
    infer_epoch = getattr(args, "infer_epoch", "final")
    
    try:
        print("a")
        const_params = ConstantParameters(inference=True, fisher_info=False, args=args)
        print("b")
        const_params.save_intermediate_layer = save_intermediate_layer
        const_compo = ConstantComponents(params=const_params, tasks=['inference'])
        print("d")
        const_compo.logger_constant.record(f"Inference initialized\n")
        const_compo.logger_constant.record(f"================================")
        print("f")
        hp_tuner = HyperparameterTuner(params=const_params, logger=const_compo.logger_constant, tasks=["inference"])
        tot_combo_num = hp_tuner.instructions.tot_combo_num
            
        for i in range(tot_combo_num):
            const_compo.logger_constant.record(f"Cycle {i}: total combo = {tot_combo_num}")
            hp_instructions = copy.copy(hp_tuner.give_instructions(i, logger=const_compo.logger_constant, params=const_params))
            hp_instructions.harmonize_instructions(logger=const_compo.logger_constant, tasks=["inference"])
            
            const_compo.logger_constant.record(f"hp_instructions: {hp_instructions}")
            params = TunableParameters(hp_instructions=hp_instructions) + const_params
            
            assert hasattr(params, "inference")
            params._set_secondary_parms()

            print(f"{params.DIRECTORIES.MODEL_DIR}")
            if os.path.exists(f"{params.DIRECTORIES.MODEL_DIR}"):
                if hp_tuner.is_exception():
                    continue

                params._write_metadata()
                
                inference_run(components=const_compo, 
                    params=params, 
                )
            else:
                const_compo.logger_constant.record(f"Model direction not found: {params.DIRECTORIES.MODEL_DIR}\n")
    
    except Exception as e:
        with open(f"{const_params.DIRECTORIES.PROJ_DIR}/error.txt", "a+") as out:
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n")
    
