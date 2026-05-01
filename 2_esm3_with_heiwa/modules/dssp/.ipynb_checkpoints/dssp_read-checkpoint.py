import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

@dataclass
class DSSPoutput:
    coordinates : torch.Tensor| None=None
    sequence : Optional[str] | None=None
    secondary_structure : Optional[str] | None=None
    sasa : Optional[List[float]] | None=None

def read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end):
    x = row[x_start:x_end].replace(" ", "")
    y = row[y_start:y_end].replace(" ", "")
    z = row[z_start:z_end].replace(" ", "")
    return x, y, z

def read_coord_coarse(row, coords, resi, x_start, x_end, y_start, y_end, z_start, z_end):
    coords[resi-1, 0, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 1, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 2, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    return coords

def get_index(title:str):
    idx = {}
    idx["Resi"] = (title.find("RESIDUE"), title.find("RESIDUE")+5)
    idx["Resn"] = (title.find("AA"), title.find("AA")+1)
    idx["SS8"] = (title.find("STRUCTURE"), title.find("STRUCTURE")+1)
    idx["ACC"] = (title.find("ACC"), title.find("ACC")+3)
    idx["Hbond0"] = (title.find("N-H-->O")-2, title.find("N-H-->O")+7)
    idx["Hbond1"] = (title.find("O-->H-N")-2, title.find("O-->H-N")+7)
    idx["Hbond2"] = (title.find("N-H-->O", title.find("N-H-->O")+1)-2, title.find("N-H-->O", title.find("N-H-->O")+1)+7)
    idx["Hbond3"] = (title.find("O-->H-N", title.find("O-->H-N")+1)-2, title.find("O-->H-N", title.find("O-->H-N")+1)+7)
    idx["X-CA"] = (title.find("X-CA")-2, title.find("X-CA")+4)
    idx["Y-CA"] = (title.find("Y-CA")-2, title.find("Y-CA")+4)
    idx["Z-CA"] = (title.find("Z-CA")-2, title.find("Z-CA")+4)
    return idx

def extractNumbers(row, idx, field):
    start, end = idx[field]
    return str(row[start:end].replace(" ", ""))

def readDSSP(filename, dssp_start, max_len, output_tracks):
    """ dssp_name is assumed to be f"AF-{entry}-F1-model_v4.dssp"
    length is the length after STP truncation
    return seq, ss8, ACC, coords 
    retain_residues is assumes STP has been cut already """
    outputs = {}
    outputs['coordinates'] = np.full((max_len, 3, 3), np.nan)
    if "sequence" in output_tracks:
        outputs["sequence"] = ""
    if "secondary_structure" in output_tracks:
        outputs["secondary_structure"] = ""
    if "sasa" in output_tracks:
        outputs["sasa"] = [None] * max_len
    # print("hello1")
    length=0
    with open(f"{filename.replace('.dssp', '')}.dssp", "r") as content:
        allrows = content.readlines()
        status = "wait"
        
        for row in allrows:
            if length >= max_len:
                return "max_len_exceeded"
            # row = row.decode("utf-8")
            if row.count(" # ") > 0:
                idx = get_index(row)
                x_start, x_end = idx["X-CA"]
                y_start, y_end = idx["Y-CA"]
                z_start, z_end = idx["Z-CA"]
                status = "read"
                # print(row)
                
            elif status == "read":
                start, end = idx["Resi"]
                
                if row[start:end].replace(" ", "").isdigit():
                    resi = int(row[start:end]) - dssp_start
                    if resi > 0:
                        # AA
                        if "sequence" in output_tracks:
                            resn = extractNumbers(row, idx, "Resn")
                            outputs["sequence"] += resn
    
                        # Coords
                        # print(row)
                        # print(resi)
                        outputs['coordinates'] = read_coord_coarse(row, outputs['coordinates'], 
                                                                   resi, 
                                                                   x_start, x_end, 
                                                                   y_start, y_end, 
                                                                   z_start, z_end)
                    
                        # SS8
                        if "secondary_structure" in output_tracks:
                            ss8 = extractNumbers(row, idx, "SS8")
                            outputs["secondary_structure"] += ss8

                        if "sasa" in output_tracks:
                            sasa = extractNumbers(row, idx, "ACC")
                            outputs["sasa"][resi-1] = float(sasa)
                        length += 1
                        
    # print("hello2")
    # outputs['coordinates'] = torch.from_numpy(outputs['coordinates'])
    # protein = ESMProtein()
    # protein.coordinates = outputs['coordinates']
    final_output = DSSPoutput(coordinates=torch.from_numpy(outputs['coordinates'][:length, ...]))
    for track in output_tracks:
        setattr(final_output, track, outputs[track][:length])
    
    return final_output
