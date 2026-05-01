from modeller import *
from modeller.automodel import *
from modeller.parallel import *
import os

env = Environ()
default_output = "target.B99990001.pdb"

def alignment_writer(sequence, length, base_name, folder_out):
    """Write alignment file for MODELLER
    Args:
        row: pandas DataFrame row containing 'Sequence_final', 'Filename', 'Y_thermo'
    """
    file_out_name = f"{folder_out}/{base_name}.ali"
    
    # Prepare the alignment content
    alignment_content = f""">P1;backbone
structureX:{base_name}.pdb:1:A:{length}:A::::
{'-' * length}*

>P1;target
sequence:target::::::::
{sequence}*
"""
    
    # Write the alignment file
    with open(file_out_name, 'w') as f:
        f.write(alignment_content)
    
    print(f"Created alignment file: {file_out_name}")
    return file_out_name


def fill_atom_to_pdb(base_name, folder_out):
    """Build full-atom model from backbone-only PDB
    Args:
        row: pandas DataFrame row containing 'Filename', 'Y_thermo'
    """
    pdb_path = f"{folder_out}/{base_name}.pdb"
    ali_file = f"{folder_out}/{base_name}.ali"
    ali_final = f"{folder_out}/{base_name}_final.ali"
    
    # Read the template structure
    aln = Alignment(env)
    mdl = Model(env, file=pdb_path.replace('.pdb', ''), model_segment=('FIRST:A', 'LAST:A'))
    aln.append_model(mdl, align_codes='backbone', atom_files=pdb_path)
    aln.append(file=ali_file, align_codes='target')
    aln.align2d()
    aln.write(file=ali_final, alignment_format='PIR')
    
    # Build the model with side chains
    a = AutoModel(env, 
                  alnfile=ali_final,
                  knowns='backbone', 
                  sequence='target')
    a.starting_model = 1
    a.ending_model = 1
    
    # Set output directory
    a.outputs = ['MODELS']  # Only output final model
    
    a.make()
    
    # Rename output to something more meaningful
    # MODELLER creates: target.B99990001.pdb
    final_output = f"{folder_out}/{base_name}_full.pdb"
    
    if os.path.exists(default_output):
        os.rename(default_output, final_output)
        print(f"Created full-atom model: {final_output}")
        for path in [pdb_path, ali_file, ali_final]:
            os.remove(path)
    else:
        print(f"Warning: Expected output {default_output} not found!")
    
    return final_output

def fill_atom_to_pdb_in_single(base_name, folder_out, sequence, length):
    alignment_writer(
        sequence=sequence, length=length, base_name=base_name, folder_out=folder_out
    )
    print("wrote alignment")
    
    fill_atom_to_pdb(
        base_name, folder_out
    )
    print("pdb filled")


