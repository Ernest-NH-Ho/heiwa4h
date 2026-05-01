import argparse
import os
import inspect
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
import traceback
import pandas as pd
from modules.file_handling import compress, decompress, ensure_dirs
from modules.train_control.parameters import ConstantParameters, TunableParameters
from modules.epoch_control.epoch_run import EpochRunner
from modules.train_control.hp_tuning import HyperparameterTuner, HyperparameterTuning_Report, HyperparameterTuner_Instruction
from modules.train_control.utils import import_data, update_dataloader_batch_size, adjust_weightPN_dim
from modules.train_control.components import ConstantComponents, TunableComponents
from modules.dataloader import compile_data_train, compile_data_single

def train_run(
    components:ConstantComponents, 
    params:TunableParameters, 
    test_epoch=None, 
    hp_instructions:HyperparameterTuner_Instruction|None=None
):
    """Main training function."""
    try:
        print("Starting training run...")

        # Initialize components
        assert hasattr(components, "tasks")
        components = TunableComponents(
            params=params, test_epoch=test_epoch, # debug=True
            hp_instructions=hp_instructions, components=components
        )
        assert hasattr(components, "tasks")
        
        components.logger_tunable.record("="*50)
        components.logger_tunable.record("TRAINING SESSION STARTED")
        components.logger_tunable.record("="*50)
        
        # Log parameters
        components.logger_tunable.record(f"Device: {params.DEVICE}")
        components.logger_tunable.record(f"Max epochs: {params.MAX_EPOCHS}")
        components.logger_tunable.record(f"Batch size: {params.BATCH_SIZE}")
        for attribute in inspect.getmembers(params):
            if not attribute[0].startswith('_'):
                components.logger_tunable.record(f"{attribute[0]}: {getattr(params, f'{attribute[0]}')}")
        
        # Create or load data
        if test_epoch != 'final':
            components.logger_tunable.record("Compiling data...")
            # Create dataloaders
            data_loader = compile_data_train(
                components.dataset_train_valid,
                params=params
            )
            components.logger_tunable.record(f"Data loaded: Train={len(data_loader['train'].dataset)}, Valid={len(data_loader['valid'].dataset)}")
        
        
        # Initialize logbook
        DF_weights = pd.DataFrame()

        # Initialize epoch runner
        epoch_runner = EpochRunner(params=params, expected_tasks=['train', 'valid', 'test'])
        updated_dropout = None
        
        if test_epoch != 'final':
            components.logger_tunable.record("Starting training loop...")
            best_epoch = 0
        
            # Training loop
            for epoch in range(hp_instructions.start_epoch, hp_instructions.end_epoch):
                if hp_instructions is not None:
                    components.logger_tunable.record(f"\n{'='*20} EPOCH {epoch+1}/{hp_instructions.end_epoch} {'='*20}")
                else:
                    components.logger_tunable.record(f"\n{'='*20} EPOCH {epoch+1}/{params.MAX_EPOCHS} {'='*20}")

                # Random seed
                random_seeds = components.rs_generator.generate(epoch)
                torch.manual_seed(random_seeds)
                components.logger_tunable.record(f"Random seed set to {random_seeds}")
                
                # Get schedule instructions
                schedule_instructions = components.train_schedule.instruct(epoch)
                current_batch_size = schedule_instructions['batch_size']

                # Get grad instructions
                components.model._set_grad_by_schedule(epoch=epoch, logger=components.logger_tunable)
                components.model._set_lr_by_schedule(epoch=epoch, logger=components.logger_tunable)
                # Update Le Chatelier bias default
                if params.TUNE_LE_CHA_BIAS:
                    components.model.update_le_cha_bias(params=params, epoch=epoch, task="train", logger=components.logger_tunable)
                
                # Update batch size if needed
                if current_batch_size != data_loader['train'].batch_size:
                    components.logger_tunable.record(f"Updating batch size from {data_loader['train'].batch_size} to {current_batch_size}")
                    data_loader['train'] = update_dataloader_batch_size(data_loader['train'], current_batch_size)
                    data_loader['valid'] = update_dataloader_batch_size(data_loader['valid'], current_batch_size)
                
                # Training phase
                components.logger_tunable.record("Running training phase...")
                components.logbook = epoch_runner.run(
                    train_valid_test='train',
                    data_source=data_loader['train'],
                    model=components.model,
                    epoch=epoch,
                    logbook=components.logbook,
                    logger=components.logger_tunable,
                    computation_monitor=schedule_instructions["computation_monitor"], 
                    params=params
                )
                
                # Validation phase
                components.logger_tunable.record("Running validation phase...")
                components.logbook = epoch_runner.run(
                    train_valid_test='valid',
                    data_source=data_loader['valid'],
                    model=components.model,
                    epoch=epoch,
                    logbook=components.logbook,
                    logger=components.logger_tunable,
                    computation_monitor=schedule_instructions["computation_monitor"], 
                    params=params
                )
                components.model.report_weight(DF_weights, f"{params.WEIGHT_RECORD}")
                
                # Extract latest metrics in case earlystop or dropout_tuning
                train_data = components.logbook.get_phase_data(epoch=epoch, phase='train')[params.EARLYSTOP_METRICS]
                valid_data = components.logbook.get_phase_data(epoch=epoch, phase='valid')[params.EARLYSTOP_METRICS]

                # Update dropout if scheduled
                if schedule_instructions['dropout_tuner']:
                    updated_dropout = components.model.update_dropout(
                        logger=components.logger_tunable, 
                        train_data=train_data[params.EARLYSTOP_METRICS].min(axis=1), 
                        valid_data=valid_data[params.EARLYSTOP_METRICS].min(axis=1), 
                        epoch=epoch, 
                    )
                    
                # Early stopping
                components.early_stop(
                    model=components.model,
                    epoch=epoch,
                    train_metrics=train_data.loc[len(train_data)-1],
                    valid_metrics=valid_data.loc[len(valid_data)-1],
                    logger=components.logger_tunable,
                    folder=params.DIRECTORIES.HP_FOLDER,
                    final=epoch==hp_instructions.end_epoch-1, 
                    hp_tuning=hp_instructions is not None
                )

                ## Save logbook
                components.logbook.save_csv(f"logbook_temp.csv", params.DIRECTORIES.MODEL_DIR)
                
                components.logbook.df.loc[(components.logbook.df["epoch"]==epoch) & (components.logbook.df["train_valid_test"]=="train"), "saved_chpt"] = components.early_stop.saved
                components.logbook.df.loc[(components.logbook.df["epoch"]==epoch) & (components.logbook.df["train_valid_test"]=="valid"), "saved_chpt"] = components.early_stop.saved
                    
                    
                if components.early_stop.ready_to_stop:
                    components.logger_tunable.record("Ready to stop. Breaking training loop.")
                    best_epoch = components.early_stop.best_epoch
                    print(f"best_epoch = {best_epoch}")
                    break
                    
                # Plot learning curve if scheduled
                if schedule_instructions['learn_curve']:
                    components.logger_tunable.record("Plotting learning curve...")
                    components.logbook.plot_LearnCurve(title=f"Learning Curve", save_dir=params.DIRECTORIES.MODEL_DIR)
                
                

            # ============================== train loop stop here ===================================================
            del data_loader
            
            # Final actions after training loop
            components.logger_tunable.record("\n" + "="*50)
            components.logger_tunable.record("TRAINING COMPLETED")
            components.logger_tunable.record("="*50)
    
            # Plot final learning curve
            components.logger_tunable.record("Plotting final learning curve...")
            components.logbook.plot_LearnCurve(title=f"Learning Curve", save_dir = f"{params.DIRECTORIES.MODEL_DIR}")
            
            # Decide and save final model
            if components.early_stop.ready_to_stop:
                components.model.decide_final(params.DIRECTORIES.HP_FOLDER, components.early_stop.best_epoch, hp_instructions.tot_combo_num==1)
        
        # Load best model and run test
        if params.TESTING:
            if test_epoch is None:
                test_epoch = int(components.early_stop.best_epoch)
            components.logger_tunable.record(f"\nRunning final test with best model from epoch {test_epoch} ...")

            # Random seed
            random_seeds = components.rs_generator.generate(test_epoch)
            torch.manual_seed(random_seeds)
            components.logger_tunable.record(f"Random seed set to {random_seeds}")
            
            # Compress the starting model
            for part in components.model.optimizer:
                compress(
                    f"{params.DIRECTORIES.MODEL_DIR}/{part}_epoch_{hp_instructions.start_epoch}_model.pth", 
                    logger=components.logger_tunable, num_cpu=params.NUM_WORKERS
                )
                compress(
                    f"{params.DIRECTORIES.MODEL_DIR}/{part}_epoch_{hp_instructions.start_epoch}_optimizer.pth", 
                    logger=components.logger_tunable, num_cpu=params.NUM_WORKERS
                )
            
            # Load the best model
            components.model._loadpretrain(
                hp_folder=params.DIRECTORIES.HP_FOLDER, epoch=test_epoch,
                task='test', device=params.DEVICE, logger=components.logger_tunable
            )
            
            # Run test
            test_loader = compile_data_single(
                target='test', 
                dataset=components.dataset_test,
                params=params
            )
            test_batch_size = min(params.BATCH_SIZE, len(test_loader.dataset))
            test_loader = update_dataloader_batch_size(test_loader, test_batch_size)
            
            components.logger_tunable.record(f"Data loaded: Test={len(test_loader.dataset)}")
            
            components.logbook = epoch_runner.run(
                train_valid_test='test',
                data_source=test_loader,
                model=components.model,
                epoch=test_epoch,
                logbook=components.logbook,
                logger=components.logger_tunable,
                params=params
            )
            
            # Log test results
            test_results = components.logbook.df[(components.logbook.df["epoch"]==test_epoch) & (components.logbook.df["train_valid_test"]=="test")].squeeze().to_dict()
            
            components.logger_tunable.record(f"Test Results using epoch {test_epoch}:")
            components.logger_tunable.record(f"  Test MCC: {test_results['mcc']:.4f}")
            components.logger_tunable.record(f"  Test Accuracy: {test_results['acc']:.4f}")
            components.logger_tunable.record(f"  Test F1 Score: {test_results['f1']:.4f}")
            components.logger_tunable.record(f"  Test Sensitivity: {test_results['sens']:.4f}")
            components.logger_tunable.record(f"  Test Precision: {test_results['prec']:.4f}")
            components.logger_tunable.record(f"  Test Specificity: {test_results['spec']:.4f}")

            # Compress the tested model
            for part in components.model.optimizer:
                compress(
                    f"{params.DIRECTORIES.MODEL_DIR}/{part}_epoch_{test_epoch}_model.pth", 
                    logger=components.logger_tunable, num_cpu=params.NUM_WORKERS
                )
                compress(
                    f"{params.DIRECTORIES.MODEL_DIR}/{part}_epoch_{test_epoch}_optimizer.pth", 
                    logger=components.logger_tunable, num_cpu=params.NUM_WORKERS
                )
        
        # Save logbook as csv
        components.logbook.save_csv(filename = "logbook.csv", save_dir=params.DIRECTORIES.MODEL_DIR)
        if os.path.exists(f"{params.DIRECTORIES.MODEL_DIR}/logbook_temp.csv"):
            os.remove(f"{params.DIRECTORIES.MODEL_DIR}/logbook_temp.csv")
            
        components.logger_tunable.record(f"\nTraining session completed successfully!")
        components.logger_tunable.record(f"Results saved to: {params.DIRECTORIES.MODEL_DIR}")
        components.logger_tunable.record("="*50)
                
        if hp_instructions is not None:
            earlystop_report = components.early_stop.export_report(epoch)
            return HyperparameterTuning_Report(logbook=components.logbook, earlystop_report=earlystop_report, 
                                               updated_dropout=updated_dropout)
        
    except Exception as e:
        with open(f"{params.DIRECTORIES.LOG_DIR}/error.txt", "a+") as out:
            out.write(f"Combination: \n{hp_instructions}\n\n")
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n")
            quit()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that takes an input value.")
    parser.add_argument("-head", type=int, required=True, help="How many implicit head")
    args = parser.parse_args()
    
    try:
        test_epoch = None
        const_params = ConstantParameters(inference=False, fisher_info=False, args=args) # 
        if os.path.exists(f"{const_params.DIRECTORIES.PROJ_DIR}/hp_tuning_combo_final.csv"):
            quit()
        if const_params.BACKUP_SSD:
            from modules.file_handling import FolderTransferer
            folder_transferer = FolderTransferer(const_params)
            
        const_compo = ConstantComponents(params=const_params, tasks=['train', 'valid', 'test'])
        const_params = adjust_weightPN_dim(params=const_params, dataset=const_compo.dataset_train_valid)
        const_compo.logger_constant.record(f"Hyperparameter tuner initialized\n")
        
        hp_tuner = HyperparameterTuner(params=const_params, tasks=['train', 'valid', 'test'], logger=const_compo.logger_constant)
        const_compo.logger_constant.record(list(hp_tuner.hp_to_try["value"].keys()))
        const_compo.logger_constant.record(hp_tuner.hp_to_try["value"])
        start_cycle = hp_tuner.find_latest_record(params=const_params, logger=const_compo.logger_constant)
        
        const_compo.logger_constant.record(f"================================")
        const_compo.logger_constant.record(f"Start trying combinations from cycle {start_cycle}")
        const_compo.logger_constant.record(f"Total number of combo: {hp_tuner.instructions.tot_combo_num}")
        const_compo.logger_constant.record(f"================================")
        while hp_tuner.instructions.tot_combo_num > 0:
            # print("main", hp_tuner.instructions.tot_combo_num)
            const_compo.logger_constant.record(f"\nCycle: {hp_tuner.instructions.numb_cycle+1}")
            const_compo.logger_constant.record(f"Total number of combo: {hp_tuner.instructions.tot_combo_num}")
            for i in range(hp_tuner.instructions.tot_combo_num):
                print(f"Cycle {i}: total combo =", hp_tuner.instructions.tot_combo_num)
                hp_instructions = copy.copy(hp_tuner.give_instructions(i, params=const_params, logger=const_compo.logger_constant))
                hp_instructions.harmonize_instructions(logger=const_compo.logger_constant, tasks=["train", "valid", "test"])
                
                const_compo.logger_constant.record(f"hp_instructions: {hp_instructions}")
                params = TunableParameters(hp_instructions=hp_instructions) + const_params
                
                assert hasattr(params, "inference")
                params._set_secondary_parms()
                
                if hp_tuner.is_exception():
                    continue

                # Move folder from another SSD back
                if params.BACKUP_SSD:
                    folder_transferer._monitor_capacity()
                    if hp_instructions.numb_cycle > 0:
                        folder_transferer._retrieve(params)

                # Check if this task is duplicated
                if os.path.exists(f"{params.DIRECTORIES.MODEL_DIR}/logbook.csv"):
                    already_done = hp_tuner.read_combo_metric_from_scratch(params=params, i=i)
                    if already_done:
                        continue
                        
                ensure_dirs(params)
                params._write_metadata()
                
                hp_tuning_report = train_run(components=const_compo, 
                    params=params, test_epoch=test_epoch, hp_instructions=hp_instructions
                )
                hp_tuner.read_combo_metric(params=params, hp_tuning_report=hp_tuning_report, i=i)
                
                if params.BACKUP_SSD:
                    if i%2 == 1 and hp_tuner.instructions.tot_combo_num>1:
                        folder_transferer._backup(params)
                        
            hp_tuner.conclude_and_record_current_cycle(params=params)
            hp_tuner.next_cycle(params=params, logger=const_compo.logger_constant)
            
    except Exception as e:
        with open(f"{const_params.DIRECTORIES.PROJ_DIR}/error.txt", "a+") as out:
            out.write(f"Combination: \n{hp_tuner.combinations}\n\n")
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n")
    
