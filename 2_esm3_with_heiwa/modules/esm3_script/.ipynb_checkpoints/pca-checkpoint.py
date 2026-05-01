import os
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor

def pca(file_name, folder, avg, sd, eigvec, remove_old_files):
    try:
        if not os.path.exists(f"{folder}_pca/{file_name.replace('.pt', '')}.pt"):
            tensor = torch.load(f"{folder}/{file_name}", \
                                map_location='cuda', weights_only=True).float()
            tensor = (tensor - avg)/(sd*1000)
            tensor = tensor @ eigvec/2
            torch.save(tensor, f"{folder}_pca/{file_name}")
            if remove_old_files:
                os.remove(f"{folder}/{file_name}")
        else:
            print(f"{file_name} already done")
                    
    except Exception as e:
        with open(f"error.txt", "a+") as out:
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n")

def pca_multiple(data_config, DF=None):
    stream = torch.cuda.Stream(device=data_config.DEVICE)
    avg = torch.load(f"{data_config.PCA_DIR}/avg1536_wt.pt", 
                     map_location='cuda', weights_only=True)
    sd = torch.load(f"{data_config.PCA_DIR}/sd1536_wt.pt", 
                    map_location='cuda', weights_only=True)
    eigvec = torch.load(f"{data_config.PCA_DIR}/eigvec.pt", 
                        map_location='cuda', weights_only=True)
    with torch.cuda.stream(stream):
        with ThreadPoolExecutor(max_workers=data_config.NUM_CPU) as executor:
            if DF==None:
                all_files = [file_name for file_name in os.listdir(f"{data_config.ESM3_EMB_DIR}") if file_name.endswith('.pt')]
            else:
                all_files = [f"{data_config.ESM3_EMB_DIR}/{filename}.pt" for file_name in DF["Filename"].unique()]
            futures = [executor.submit(pca, 
                                       file_name=file_name, 
                                       folder=data_config.ESM3_EMB_DIR, 
                                       avg=avg, sd=sd, eigvec=eigvec, 
                                       remove_old_files=data_config.REMOVE_OLD_EMB) for file_name in all_files]
            for fut in futures:
                fut.result()