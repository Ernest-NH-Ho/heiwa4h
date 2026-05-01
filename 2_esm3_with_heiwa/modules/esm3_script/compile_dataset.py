import os
import pandas as pd
import torch

def compile_dataset_wif_pooling(data_config, DF=None):
    if data_config.MODE == "naive":
        all_emb_filenames = pd.Series([file.replace('.pt', '') for file in list(os.listdir(f"{data_config.ESM3_EMB_DIR}_pca")) if file.endswith('.pt')])
    elif data_config.MODE == "cleavage":
        all_emb_filenames = pd.Series(DF["Filename"])
    print(len(all_emb_filenames))
    print(all_emb_filenames)
    X = torch.zeros((len(all_emb_filenames), 3, 1536), 
                    dtype=torch.float32, device='cuda')
    def make_X(row):
        print('load emb')
        embed = torch.load(f"{data_config.ESM3_EMB_DIR}_pca/{row['emb_filenames'].replace('.pt', '')}.pt", 
                           map_location='cuda', weights_only=True)
        X[row.name, 0, ...] = embed[0, ...]
        X[row.name, -1, ...] = embed[-1, ...]
        X[row.name, 1, ...] = embed[1:-1, ...].mean(dim=0)

    DF = pd.DataFrame()
    DF['emb_filenames'] = all_emb_filenames
    DF[['emb_filenames']].apply(lambda x:make_X(x), 
                                axis=1)
    torch.save(X, f"{data_config.DATA_DIR}/{data_config.INFER_FILENAME.replace('.pt', '')}.pt")
    if data_config.MODE == "naive":
        # Keep a record of the file name in order
        
        DF.to_csv(f"data_csv/{self.DATASET_NAME}/{data_config.INFER_FILENAME.replace('.csv', '')}.csv", 
                  index=False)