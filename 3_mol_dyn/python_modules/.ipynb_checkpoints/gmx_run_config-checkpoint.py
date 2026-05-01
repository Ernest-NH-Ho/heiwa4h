
class Gmx_Scripts:
    def __init__(self):
        pass

    def pdb2gmx(self, water_model):
        

class Gmx_Config:
    def __init__(self, csv_path):
        self.data = pd.read_csv(f"{csv_path.replace('.csv', '')}.csv", header=0)
        