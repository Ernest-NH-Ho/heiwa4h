import os
import torch
import pickle
import pandas as pd
import traceback
# from multiprocessing import Pool
# import multiprocessing as mp
from esm.sdk.api import GenerationConfig, ESMProtein
from modules.dssp.dssp_read import readDSSP
from modules.modeller_script import fill_atom_to_pdb_in_single
# from modules.pdb_fixer import fix_pdb_full_atom

# mp.set_start_method('spawn', force=True)

def esm3_generation(data_config, protein, model, heiwa_model, heiwa_normalizer, seq_prompt, out_filename):
    if protein == "max_len_exceeded":
        pass
    tracks = data_config.ESM3_TRACKS
    fname = None
    for i, track in enumerate(tracks):
        # print(track)
        if i == len(tracks)-1:
            fname = out_filename
            
        track_k = list(track.keys())[0]
        track_v = list(track.values())[0]
        # print(track)
        # print(fname)
        condition_on_coordinates_only = True if track_k == 'structure' else False
        
        if track_k=='sequence':
            protein.sequence = seq_prompt
            # print(seq_prompt)
        elif track_k=='structure' and track_v == 8:
            protein = model.generate(protein, GenerationConfig(track=track_k, num_steps=track_v, \
                                                               temperature=data_config.ESM3_TEMP, 
                                                               temperature_annealing=True, \
                                                               condition_on_coordinates_only = condition_on_coordinates_only), \
                                     filename=fname, 
                                     layer_i=-1, 
                                     heiwa_model=heiwa_model, 
                                     heiwa_normalizer=heiwa_normalizer)
            continue
        protein = model.generate(protein, GenerationConfig(track=track_k, num_steps=track_v, \
                                                           temperature=data_config.ESM3_TEMP, 
                                                           temperature_annealing=True, \
                                                           condition_on_coordinates_only = condition_on_coordinates_only), \
                                 filename=fname, 
                                 layer_i=-1)
    return protein
    
def check_generation_success(data_config, DF):
    """Only applicable to cleavage mode. For updating the csv file"""
    DF["generation_success"] = DF["Filename"].apply(lambda x:os.path.exists(f"{data_config.ESM3_PROT_DIR}/{x.replace('.dssp', '')}.pkl"))
    DF[DF["generation_success"]==False].to_csv(f"{data_config.CSV_PATH.replace('.csv', '')}_generateFail.csv", 
                                              index=False)
    DF = DF[DF["generation_success"]==True].reset_index(drop=True)
    DF.to_csv(f"{data_config.CSV_PATH.replace('.csv', '')}_generated.csv", 
              index=False)
    return DF

def save_protein(protein, out_filename):
    with open(f"{out_filename.replace('.pkl', '')}.pkl", 'wb') as file:
        pickle.dump(protein, file)

def generate_proteins(data_config, use_heiwa:bool):
    stream = torch.cuda.Stream(device=data_config.DEVICE)
    model = torch.load(f"{data_config.ESM3_PATH}", 
                       map_location=data_config.DEVICE, 
                       weights_only=False)
    print('model loaded')

    # Load HEIWA model
    heiwa_model = None
    heiwa_normalizer = None
    if use_heiwa:
        from modules.heiwa.model import HEIWA3H
        from modules.heiwa.normalizer import HEIWA_normalizer
        heiwa_model = HEIWA3H()
        heiwa_model.to("cuda")
        heiwa_normalizer = HEIWA_normalizer()
        print('heiwa loaded')
        
    if data_config.MODE == "naive":
        # print('naive1')
        for file in os.listdir(f"{data_config.DSSP_DIR}"):
            base_name = f"{file.replace('.dssp', '')}_heiwa{int(use_heiwa)}"
            
            if not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}") and \
            not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb") and \
            not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}_full.pdb") and \
            os.path.exists(f"{data_config.DSSP_DIR}/{DF.loc[i, 'Filename']}.dssp"):
                
                dssp_output = readDSSP(filename=f"{data_config.DSSP_DIR}/{file}", 
                                   dssp_start=0, 
                                   max_len=data_config.MAX_LEN, 
                                   output_tracks=data_config.DSSP_OUTPUT)

                protein = ESMProtein()
                for track in ['sequence', 'coordinates']:
                    setattr(protein, track, getattr(dssp_output, track))
                
                # print(protein)
                protein = esm3_generation(data_config=data_config, 
                                          protein=protein, 
                                          model=model, heiwa_model=heiwa_model, heiwa_normalizer=heiwa_normalizer, 
                                          seq_prompt=DF.loc[i, 'Sequence_final_prompt'], 
                                          out_filename=f"{data_config.ESM3_EMB_DIR}/{base_name}")
                save_protein(protein=protein, 
                             out_filename=f"{data_config.ESM3_PROT_DIR}/{base_name}")
                protein.to_pdb(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb")

                # fix_pdb_full_atom(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb")
                fill_atom_to_pdb_in_single(
                    base_name=base_name, 
                    folder_out=f"{data_config.ESM3_PROT_DIR}", 
                    sequence=protein.sequence, 
                    length=len(protein.sequence)
                )

        del model

        # def fill_pdb_atoms_single(file):
        #     os.chdir(f"tmp{i%data_config.NUM_CPU}")
        #     base_name = f"{file.replace('.dssp', '')}_heiwa{int(use_heiwa)}"
        #     with open(f"../{data_config.ESM3_PROT_DIR}/{base_name}.pkl", "rb") as file:
        #         protein = pickle.load(file)
                
            
        
        # with Pool(processes=data_config.NUM_CPU) as pool:
        #     for i in range(data_config.NUM_CPU):
        #         os.makedirs(f"tmp{i}", exist_ok=True)
        #     tasks = [file for file in os.listdir(f"{data_config.DSSP_DIR}")]
        #     pool.map(fill_pdb_atoms_single, tasks)
            
    elif data_config.MODE == "cleavage":
        import pandas as pd
        DF = pd.read_csv(f"{data_config.CSV_PATH}", header=0)
        DF["DSSP_start"] = DF["DSSP_start"].astype(int)
        
        for i, row in DF.iterrows():
            base_name = f"{row['Filename'].replace('.dssp', '')}_heiwa{int(use_heiwa)}"
            if not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}.pkl") and \
            not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb") and \
            not os.path.exists(f"{data_config.ESM3_PROT_DIR}/{base_name}_full.pdb") and \
            os.path.exists(f"{data_config.DSSP_DIR}/{row['Filename']}.dssp"):
                
                # print(DF.loc[i, 'Filename'])
                dssp_output = readDSSP(filename=f"{data_config.DSSP_DIR}/{row['Filename']}", 
                                   dssp_start=row["DSSP_start"], 
                                   max_len=data_config.MAX_LEN, 
                                   output_tracks=data_config.DSSP_OUTPUT)
                # print(dssp_output)
                protein = ESMProtein()
                for track in ['sequence', 'coordinates']:
                    setattr(protein, track, getattr(dssp_output, track))
                    
                # print(protein)
                if protein =="max_len_exceeded":
                    continue
                # print("after read dssp", hasattr(protein, "coordinate"))
                protein = esm3_generation(data_config=data_config, 
                                          protein=protein, 
                                          model=model, heiwa_model=heiwa_model, heiwa_normalizer=heiwa_normalizer, 
                                          seq_prompt=row['Sequence_final_prompt'], 
                                          out_filename=f"{data_config.ESM3_EMB_DIR}/{base_name}")
                save_protein(protein=protein, 
                             out_filename=f"{data_config.ESM3_PROT_DIR}/{base_name}")
                protein.to_pdb(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb")

                # fix_pdb_full_atom(f"{data_config.ESM3_PROT_DIR}/{base_name}.pdb")
                
                fill_atom_to_pdb_in_single(
                    base_name=base_name, 
                    folder_out=f"{data_config.ESM3_PROT_DIR}", 
                    sequence=protein.sequence, 
                    length=row['Length_final']
                )
               
        del model
        print(f"check_generation_success")
        DF = check_generation_success(data_config=data_config, 
                                      DF=DF)

            
        return DF
