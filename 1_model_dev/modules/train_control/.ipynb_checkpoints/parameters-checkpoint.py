import os
import json
import torch
import datetime
import pickle

class Directories:
    def __init__(self):
        pass

    def __str__(self):
        cout = "All directories:\n"
        for name, value in vars(self).items():
            cout += f"{name}: {value}\n"
        return cout

class Parameters:
    def __init__(self, debug=False):
        self.debug = debug
        pass

    def __add__(self, other:"Parameters"):
        for name, value in other.__dict__.items():
            setattr(self, name, value)
        return self

    def save(path):
        with open(f"{path.replace('.pkl', '')}.pkl", "wb") as file:
            pickle.dump(self, file)

class TunableParameters(Parameters):
    """ Subject to hyperparameter tuning """
    def __init__(self, hp_instructions=None):
        super().__init__()
        self.now_tunable = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        # ====================Model parameters=================================
        if hp_instructions is not None:
            for k,v in hp_instructions.hp_instructions.to_dict().items():
                setattr(self, k, hp_instructions.dtype[k](v))
        
    def _set_secondary_parms(self):
        
        # Directories
        # HEIWA_Imp
        self.DIRECTORIES.MODEL_DIR = f"{self.DIRECTORIES.PROJ_DIR}/esm3_SelfGate_bs{self.BATCH_SIZE*self.ACCUM_STEP}_dr{str(self.DROPOUT['low'])[2:]}_{str(self.DROPOUT['high'])[2:]}_v{self.heiwa_imp_v_dim}_v_compress{self.hyb_v_compress}_Imp_hd{self.HEIWA_IMP_NUM_SOLVENT_HEAD}" #    #### 1 MixV_trainedMixer_
        
        self.DIRECTORIES.LOG_DIR = self.DIRECTORIES.MODEL_DIR
        self.DIRECTORIES.HP_FOLDER = self.DIRECTORIES.MODEL_DIR 
        self.WEIGHT_RECORD = os.path.join(self.DIRECTORIES.LOG_DIR, "model_weights.csv")
        if self.debug:
            print(f"{self.DIRECTORIES.MODEL_DIR} exists == {os.path.exists(self.DIRECTORIES.MODEL_DIR)}")
        
    def _write_metadata(self):
        print(f"metadata: {self.DIRECTORIES.LOG_DIR}/meta_{self.now_tunable}_{self.MODEL_FOLDER_NAME_BASE}.txt")
        with open(f"{self.DIRECTORIES.LOG_DIR}/meta_{self.now_tunable}_{self.MODEL_FOLDER_NAME_BASE}.txt", 'a+') as file:
            file.write(f"Time: {self.now_tunable}\n")
            file.write(f"Operator: {self.operator}\n")
            file.write(f"\n")
            file.write(f"Project purpose: {self.project_purpose}\n")
            file.write(f"Session purpose: {self.session_purpose}\n")
            file.write(f"\n")
            file.write(f"Inference: {self.inference}\n")
            print("metadata under tunable parameters written")
        if self.debug:
            input()

        
class ConstantParameters(Parameters):
    """ Not affect by hp_tuner """
    def __init__(self, inference:bool, fisher_info:bool=False, args=None):
        super().__init__()
        self.DEBUG = False
        self.inference = inference
        self.fisher_info = fisher_info
        self.save_intermediate_layer = False
        self.now_constant = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.DIRECTORIES = Directories()
        
        # Meta data
        if args!=None:
            print("input args into Constant Parameters")
            self.HEIWA_IMP_NUM_SOLVENT_HEAD = args.head

        # Project data
        self.DATA_FOLDER = 'HEIWA3P'
        self.MODEL_FOLDER_NAME = f"HEIWA_SelfGate_rot1_angAtten1_exhd{self.HEIWA_IMP_NUM_SOLVENT_HEAD}_Imp_newSASAgateSoft"
        start = self.MODEL_FOLDER_NAME.rfind('/')+1 if self.MODEL_FOLDER_NAME.count('/') > 0 else 0
        self.MODEL_FOLDER_NAME_BASE = self.MODEL_FOLDER_NAME[start:] #### 2
        print("self.MODEL_FOLDER_NAME_BASE", self.MODEL_FOLDER_NAME_BASE)
        self.project_purpose = 'Supervised Training on HEIWA3P (H2O + Na + Cl)'
        self.session_purpose = 'Optimize V_dim, V_compression, sigmoid-relu ratio, Swish expansion, SASAgate without torch.where is used'
        self.operator = 'Anonymous Author'

        # Device
        self.DEVICE = 'cuda'
        self.NUM_WORKERS = 24

        # Backup on another SSD
        if not inference:
            self.BLOCK_FACTOR = 4096
            self.BACKUP_SSD = False
            if self.BACKUP_SSD:
                self.DIRECTORIES.BACKUP_SSD_ADDRESS = "/media/user/dirty-drive"
                self.DIRECTORIES.MAIN_SSD_ADDRESS = "/"
                self.DIRECTORIES.BACKUP_ADDRESS = f"{self.BACKUP_SSD_ADDRESS}/AnonymousAuthor/Proj2026_HEIWA/models/{self.MODEL_FOLDER_NAME}"
                os.makedirs(self.DIRECTORIES.BACKUP_ADDRESS, exist_ok=True)
                assert os.path.exists(self.DIRECTORIES.BACKUP_SSD_ADDRESS)
                assert os.path.exists(self.DIRECTORIES.MAIN_SSD_ADDRESS)
                assert os.path.exists(self.DIRECTORIES.BACKUP_ADDRESS)
            
            
            # Weights
            self.WEIGHT_P = {}
            self.WEIGHT_N = {}
            self.ADJUST_N = 1
        
        # Dataset parameters
        self.DATAFILE_PREFIX = ['X', 'Y', 'M', 'W0', 'L']
        self.DATAFILE_PREFIX_MODEL_INPUT = ['X', 'M', 'L']
        self.DATAFILE_PREFIX_EVAL_INPUT = ['Y', 'W0']
        if self.inference and not self.fisher_info:
            for target in ['W0', 'Y']:
                if target in self.DATAFILE_PREFIX:
                    self.DATAFILE_PREFIX.remove(target)

         # ==================== Model parameters =================================
        self.PROT_DIM = 1536
        self.DROPOUT = {'low': 0.1, 'high':0.3}

        # Salt conc
        self.TUNE_LE_CHA_BIAS = False
        self.SALT_CONC = {"water":500/9, "NA":0.15, "CL":0.15} # 
        self.LE_CHA_BIAS_INIT = -0.05 if not inference and not fisher_info else -100
        self.LE_CHA_BIAS_STEP = -0.1
        
        # HEIWA_3P Implicit
        self.HEIWA_IMP_QK_DIM = 128
        self.HEIWA_IMP_Z_QK_DIM_RATIO = 5.
        self.HEIWA_IMP_NUM_PROT_HEAD = 3
        # self.HEIWA_IMP_V_DIM = 192
        self.HEIWA_IMP_WITH_SCALER = True
        self.HEIWA_IMP_SIGMOID_FACTOR = 0
        if args.head == 4:
            self.HEIWA_IMP_HEAD_LABEL = ["water", "water", "water", "water"]
        else:
            self.HEIWA_IMP_HEAD_LABEL = ["water", "water", "water", "NA", "CL"]
        self.HEIWA_IMP_HEAD_LABEL = self.HEIWA_IMP_HEAD_LABEL[:args.head]
        self.HEIWA_IMP_HEAD_REQUIRE_ROT = [True for _ in self.HEIWA_IMP_HEAD_LABEL] # 
        self.HEIWA_IMP_ROT_HEAD_IDX = list(range(len(self.HEIWA_IMP_HEAD_REQUIRE_ROT)))
        self.HEIWA_IMP_NUM_SOLVENT_HEAD = len(self.HEIWA_IMP_HEAD_LABEL)
        self.HEIWA_IMP_LECHA_BIAS = torch.tensor([self.SALT_CONC[solv] for solv in self.HEIWA_IMP_HEAD_LABEL], dtype=torch.float32, device="cuda")
        self.HEIWA_IMP_LECHA_BIAS = torch.log(self.HEIWA_IMP_LECHA_BIAS) / 16
        self.NUM_ROT_HEAD = self.HEIWA_IMP_HEAD_REQUIRE_ROT.count(True)

        # HEIWA_3P Hybridizer 
        # self.HYB_V_COMPRESS = 0.8
        self.HYB_SWISH_EXPAND = 4.
        
        # TIM parameters
        self.TIM_MID_DIM = 6
        self.TIM_OUT_DIM = 1

        # SASA
        self.SASA_INTERMEDIATE_DIM = 1024

        # Thermo
        self.THERMO_INTERMEDIATE_DIM = 96
        self.THERMO_OUT_DIM = 1

        # ==================== Training parameters =================================
        self.LR_DECAY = 0.95 # exponential decay. You can still design your own lr warm up schedule using lr_schedule.csv
        self.MAX_EPOCHS = 2000
        self.EPOCH_FINAL = 'final'
        self.MAX_NORM = 1.0
        self.GRAD_RECORD_DIM = 5

        # Epoch parameters
        self.BATCH_SIZE = 16
        if self.fisher_info:
            self.BATCH_SIZE = 1
        self.ACCUM_STEP = 4
        self.COMPUTE_MONITOR_FREQ_BATCH = 50 * self.ACCUM_STEP
        self.INTRA_MODEL_LOSS_FACTOR = 0.1

        # Early stopping parameters
        self.metric_keys = ['mcc', 'acc', 'f1', 'prec', 'sens', 'spec', 'tp', 'tn', 'fp', 'fn'] # no need to write loss, dropout
        self.INIT_PATIENCE = 20
        self.PATIENCE = 8
        self.MIN_DELTA = 0.001
        self.SLIGHT_OVERFIT_THRES = 0.045
        self.EARLYSTOP_METRICS = ['sens', 'prec', 'spec'] #  Model performance will calculated as the min(self.EARLYSTOP_METRICS) if self.EARLYSTOP_MODE=="max"
        self.EARLYSTOP_MODE = 'max' # may change to 'min' to use 'loss' as the only EARLYSTOP_METRICS
        
        # Logging
        self.VERBOSE = True

        # Testing or not?
        self.TESTING = True
        
        # Directories
        self.DIRECTORIES.BASE_DIR = "."
        self.DIRECTORIES.DATA_PATH = f"{self.DIRECTORIES.BASE_DIR}/data/{self.DATA_FOLDER}"
        self.DIRECTORIES.PROJ_DIR = f"{self.DIRECTORIES.BASE_DIR}/models/{self.MODEL_FOLDER_NAME}"

        # Weights
        with torch.no_grad():
            if not self.inference or self.fisher_info:
                self.WEIGHT_P['train'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wp_train.pt", 
                                               map_location='cuda', weights_only=True)
                self.WEIGHT_N['train'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wn_train.pt", 
                                               map_location='cuda', weights_only=True)
            if not self.inference and not self.fisher_info:
                self.WEIGHT_P['valid'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wp_valid.pt", 
                                               map_location='cuda', weights_only=True)
                self.WEIGHT_N['valid'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wn_valid.pt", 
                                               map_location='cuda', weights_only=True)
                self.WEIGHT_P['test'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wp_test.pt", 
                                               map_location='cuda', weights_only=True)
                self.WEIGHT_N['test'] = torch.load(f"{self.DIRECTORIES.DATA_PATH}/Wn_test.pt", 
                                               map_location='cuda', weights_only=True)

    
    def _write_metadata(self):
        with open(f"{self.DIRECTORIES.LOG_DIR}/meta_{self.now_constant}_{self.MODEL_FOLDER_NAME_BASE}.txt", 'a+') as file:
            file.write(f"Time: {self.now_constant}\n")
            file.write(f"Operator: {self.operator}\n")
            file.write(f"\n")
            file.write(f"Project purpose: {self.project_purpose}\n")
            file.write(f"Session purpose: {self.session_purpose}\n")
            file.write(f"\n")
            file.write(f"Inference: {self.inference}\n")
            print("metadata under constant parameters written")
        if self.debug:
            input()
