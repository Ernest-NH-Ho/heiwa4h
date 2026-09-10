"""
GROMACS Runner Module
Orchestrates GROMACS simulations for multiple protein systems.
"""
import os
import shutil
import subprocess
import pandas as pd
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from python_modules.gromacs_script_writer import GromacsScriptWriter

edr_label = {
    "Temperature":"Temperature", 
    "Potential": "Potential", 
    "Kinetic-En.": "KE", 
    "Pressure": "Pressure", 
    "Density": "density", 
}

xtc_target = {
    "rmsf": "'C-alpha\n'", 
    "rms": "'C-alpha\nC-alpha\n'", 
    "gyrate": "'\"C-alpha\"\n'", 
    # "density": "SOL", 
}

class GromacsRunner:
    """Manages and executes GROMACS simulations for multiple protein systems."""
    
    def __init__(self, task, **kwargs):
        """ Initialize the GROMACS runner.
        Args:
            client_list: pandas DataFrame with columns: 'folder', 'temp', 'tot_time', 'add_H', 'md'
            ff: Force field name (default: 'charmm36-jul2022')
            cation: Cation name for neutralization (default: 'NA')
            anion: Anion name for neutralization (default: 'CL')
            salt_conc: Salt concentration in M (default: 0.15)
            water_model: Water model (default: 'tip3p')
            water_coord: Water coordinate file (default: 'spc216.gro')
            box_shape: Simulation box shape (default: 'dodecahedron')
            replicate: Number of independent MD runs (default: 3)
            dt: MD timestep in ps (default: 0.002)
            dt_em: Energy minimization step size (default: 0.01)
            em_max_steps: Max energy minimization steps (default: 50000)
            snapshot_interval: Output frequency for production MD (default: 5000)
            n_threads: Number of CPU threads (default: 24)
            gpu_id: GPU device ID (default: 0)
        """
        self.task = task
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Validate DataFrame
        required_cols = ['folder', 'file']
        if self.task == 'md':
            required_cols += ["temp_start", "temp_end"] # , 'tot_time'
        missing_cols = [col for col in required_cols if col not in self.client_list.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in client_list: {missing_cols}")

    
    def _run_command(self, cmd, cwd=None, input_text=None):
        """
        Execute a shell command and capture output.
        
        Args:
            cmd: Command string or list of arguments
            cwd: Working directory for command execution
            input_text: Text to pipe to stdin
            
        Returns:
            subprocess.CompletedProcess object
        """
        try:
            result = subprocess.run(
                cmd,
                shell=True if isinstance(cmd, str) else False,
                cwd=cwd,
                input=input_text,
                text=True,
                capture_output=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command failed with exit code {e.returncode}")
            print(f"Command: {cmd}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

    def plot(self, 
             name_of_run, 
             tpr_name, 
             folder_path, 
             edr_target_key=["Temperature", "Potential", "Kinetic-En.", "Pressure", "Density"], 
             xtc_target_key=["rmsf", "rms", "gyrate"], # "density"
             trj_type="xtc", 
             center=True, 
            ):

        def plot_single_energy(target, tpr_name):
            if not os.path.exists(f"{folder_path}/{tpr_name}_{edr_label[target].lower()}.xvg"):
                print(f"Calculating {target.lower()} for {name_of_run} MD...")
                cmd = f"printf '{target}\n' | gmx energy -f {tpr_name}.edr -o {tpr_name}_{edr_label[target].lower()}.xvg"
                self._run_command(cmd, cwd=folder_path)
                print(f"✓ {target} saved for {name_of_run} MD\n")
            
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(plot_single_energy, target, tpr_name) for target in edr_target_key]
            for fut in futures:
                fut.result()

        suffix=""
        if center and trj_type=="xtc":
            suffix="_pbc_center_compact"
        def plot_single_xtc(target, tpr_name):
            if not os.path.exists(f"{folder_path}/{tpr_name}_{target}.xvg"):
                print(f"Calculating {target} for {name_of_run} MD...")
                # 3 means Ca only
                cmd = f"printf {xtc_target[target]} | gmx {target} -f {tpr_name}{suffix}.{trj_type} -s {tpr_name}.tpr -o {tpr_name}_{target}.xvg"
                self._run_command(cmd, cwd=folder_path)
                print(f"✓ {target} saved for {name_of_run} MD\n")
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(plot_single_xtc, target, tpr_name) for target in xtc_target_key]
            for fut in futures:
                fut.result()
            
                
    def _get_pdb_file(self, folder_path, file):
        """Find the PDB file in the given folder."""
        pdb_file = folder_path / file
        return pdb_file
    
    def _add_hydrogen_only(self, folder_path, file, temp):
        """
        Run pdb2gmx to add hydrogens and convert back to PDB.
        
        Args:
            folder: Folder name containing the PDB file
            temp: Temperature (not used here, but kept for consistency)
        """
        pdb_file = self._get_pdb_file(folder_path, file)
        prefix = pdb_file.stem  # filename without extension
        
        print(f"\n{'='*60}")
        print(f"Adding hydrogens to: {pdb_file.name}")
        print(f"{'='*60}\n")
        
        # Step 1: Run pdb2gmx to add hydrogens
        gro_output = folder_path / f"{prefix}.gro"
        if not os.path.exists(gro_output):
            cmd = (f"gmx pdb2gmx -f {pdb_file.name} -o {prefix}.gro "
                   f"-water {self.water_model} -ff {self.ff} -ter -ignh")
            
            print(f"Running: pdb2gmx for {pdb_file.name}")
            self._run_command(cmd, cwd=folder_path)
        
        # Step 2: Convert GRO back to PDB
        pdb_h_output = folder_path / f"{prefix}_H.pdb"
        if not os.path.exists(pdb_h_output):
            cmd = f"gmx editconf -f {prefix}.gro -o {prefix}_H.pdb"
            
            print(f"Converting to PDB: {pdb_h_output.name}")
            self._run_command(cmd, cwd=folder_path)
            
            print(f"✓ Hydrogenated PDB saved: {pdb_h_output}")
            print(f"{'='*60}\n")

    def _rename_and_remove_topol(self, folder_path, new_topol_name):
        if os.path.exists(f"{folder_path}/topol.top"):
            shutil.copy(f"{folder_path}/topol.top", f"{folder_path}/{new_topol_name}.top")
            if os.path.exists(f"{folder_path}/#topol.top.1#"):
                os.remove(f"{folder_path}/#topol.top.1#")

    def concat_trj(self, folder_path, tpr_name):
        d = {
            "xtc": "trjcat", 
            "edr": "eneconv"
        }
        
        for suffix in d:
            i = 2
            file = f"{folder_path}/{tpr_name}.part{i:04}.{suffix}"
            while os.path.exists(file):
                i += 1
                file = f"{folder_path}/{tpr_name}.part{i:04}.{suffix}"
    
            if i >= 2+1:
                i_list = range(2, i)
                file_list = [f"{tpr_name}.{suffix}"] + [f"{tpr_name}.part{i:04}.{suffix}" for i in i_list]
                cmd = f"gmx {d[suffix]} -f {' '.join(file_list)} -o {tpr_name}_combine.{suffix}"
                self._run_command(cmd, cwd=folder_path)
                os.rename(f"{folder_path}/{tpr_name}.{suffix}", f"{folder_path}/{tpr_name}_old.{suffix}")
                os.rename(f"{folder_path}/{tpr_name}_combine.{suffix}", f"{folder_path}/{tpr_name}.{suffix}")

    def save_snapshot(self, folder_path, tpr_name, idx):
        if not os.path.exists(f"{tpr_name}_pbc_center_compact_dump{idx}.pdb"):
            # 0 means output the whole system
            cmd = (f"printf 'System\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}_pbc_center_compact.xtc " 
                   f" -o {tpr_name}_pbc_center_compact_dump{idx}.pdb -dump {idx}")
            self._run_command(cmd, cwd=folder_path)
        if not os.path.exists(f"{tpr_name}_pbc_center_compact_dump{idx}_Protein.pdb"):
            cmd = (f"printf 'Protein\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}_pbc_center_compact.xtc " 
                   f" -o {tpr_name}_pbc_center_compact_dump{idx}_Protein.pdb -dump {idx}")
            self._run_command(cmd, cwd=folder_path)

    def _run_preparation_steps(self, folder_path, file):
        """
        Run system preparation: pdb2gmx, create box, solvate, add ions.
        
        Args:
            folder: Folder name containing the PDB file
        """
        pdb_file = self._get_pdb_file(folder_path, file)
        prefix = pdb_file.stem
        
        print(f"\n{'='*60}")
        print(f"PREPARATION STEPS for: {pdb_file.name}")
        print(f"{'='*60}\n")
        
        # Step 1: Generate topology
        print(f"Step 1: Generating topology with pdb2gmx ...")
        if not os.path.exists(f"{folder_path}/{prefix}.gro"):
            cmd = (f"gmx pdb2gmx -f {pdb_file.name} -o {prefix}.gro "
                   f"-water {self.water_model} -ff {self.ff} -ter -ignh")
            self._run_command(cmd, cwd=folder_path)
            # print(os.listdir(folder_path))
            print("✓ Topology generated\n")
        gro_filename = f"{prefix}"
        
        # Step 2: Create box
        print(f"Step 2: Creating {self.box_shape} box with {self.box_margin} margin for {prefix}...")
        if not os.path.exists(f"{folder_path}/{gro_filename}_{self.box_shape}_{self.box_margin}.gro"):
            cmd = (f"gmx editconf -f {gro_filename}.gro -o {gro_filename}_{self.box_shape}_{self.box_margin}.gro "
                   f" -c -d {self.box_margin} -bt {self.box_shape}")
            self._run_command(cmd, cwd=folder_path)
            # print(os.listdir(folder_path))
            self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"topol_{self.box_shape}_{self.box_margin}")
            print("✓ Box created\n")
        top_filename = f"topol_{self.box_shape}_{self.box_margin}"
        gro_filename = f"{gro_filename}_{self.box_shape}_{self.box_margin}"
            
        
        # Step 3: Solvate
        print(f"Step 3: Solvating the system for {prefix}...")
        if not os.path.exists(f"{folder_path}/{gro_filename}_{self.water_model}.gro"):
            cmd = (f"gmx solvate -cp {gro_filename}.gro -cs {self.water_coord} "
                   f"-o {gro_filename}_{self.water_model}.gro -p topol.top")
            self._run_command(cmd, cwd=folder_path)
            # print(os.listdir(folder_path))
            self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_{self.water_model}")
            print("✓ System solvated\n")
        top_filename = f"{top_filename}_{self.water_model}"
        gro_filename = f"{gro_filename}_{self.water_model}"
        if self.task == 'solvate':
            return
            
        
        # Step 4: Add ions - first generate ions.tpr
        print(f"Step 4: Generating ions.tpr for {prefix}...")
        if not os.path.exists(f"{folder_path}/ions.tpr"):
            cmd = (f"gmx grompp -f ions.mdp -c {gro_filename}.gro "
                   f"-p topol.top -o ions.tpr")
            self._run_command(cmd, cwd=folder_path)
        
        print(f"Step 4b: Adding counter ions for {prefix}...")
        if not os.path.exists(f"{folder_path}/{gro_filename}_{self.cation}_{self.anion}_{self.salt_conc}.gro"):
            # 13 means SOL
            cmd = (f"printf '13\n' | gmx genion -s ions.tpr -o {gro_filename}_{self.cation}_{self.anion}_{self.salt_conc}.gro " # select a group of solvent: 13 mans SOL
                   f"-p topol.top -pname {self.cation} -nname {self.anion} "
                   f"-neutral -conc {self.salt_conc}")
            self._run_command(cmd, cwd=folder_path)
            self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_{self.cation}_{self.anion}_{self.salt_conc}")
            print(f"✓ Ions added (concentration: {self.salt_conc} M)\n")
        top_filename = f"{top_filename}_{self.cation}_{self.anion}_{self.salt_conc}"
        gro_filename = f"{gro_filename}_{self.cation}_{self.anion}_{self.salt_conc}"
        
        print(f"{'='*60}")
        print(f"PREPARATION COMPLETE for {prefix}")
        print(f"{'='*60}\n")
        return gro_filename, top_filename
    
    def _run_energy_minimization(self, folder_path, gro_filename, top_filename, i=0):
        """
        Run energy minimization.
        
        Args:
            folder: Folder name
        """
        print(f"\n{'='*60}")
        print(f"ENERGY MINIMIZATION {i} for {folder_path}")
        print(f"{'='*60}\n")
        
        # Generate em.tpr
        if not os.path.exists(f"{folder_path}/em{i}.tpr"):
            cmd = (f"gmx grompp -f minim.mdp -c {gro_filename}.gro " 
                   f"-p topol.top -o em{i}.tpr") # 
            self._run_command(cmd, cwd=folder_path)

        if not os.path.exists(f"{folder_path}/em{i}.gro"):
            # Run energy minimization
            cmd = f"gmx mdrun -v -deffnm em{i} -nt {self.n_threads} -nb gpu"
            print(f"Running energy minimization...")
            self._run_command(cmd, cwd=folder_path)
            self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_em{i}")
            print(f"✓ Energy minimization complete\n")
        
        # Generate potential energy plot
        if not os.path.exists(f"{folder_path}/em{i}_potential.xvg"):
            print("Extracting potential energy...")
            cmd = f"printf 'Potential\n' | gmx energy -f em{i}.edr -o em{i}_potential.xvg"  # 11 means potential
            self._run_command(cmd, cwd=folder_path)
            print("✓ Potential energy saved to potential.xvg\n")
            
        top_filename = f"{top_filename}_em{i}"
        gro_filename = f"em{i}"
        return gro_filename, top_filename
    
    def _run_nvt_equilibration(self, folder_path, gro_filename, top_filename, temp, i=0):
        """
        Run NVT equilibration (constant N, V, T).
        
        Args:
            folder: Folder name
            temp: Temperature in Kelvin
        """
        print(f"\n{'='*60}")
        print(f"NVT EQUILIBRATION {i} (T = {temp} K) for {folder_path}")
        print(f"{'='*60}\n")

        os.makedirs(f"{folder_path}/posre_nvt", exist_ok=True)
            
        if os.path.exists(f"{folder_path}/posre.itp"):
            os.remove(f"{folder_path}/posre.itp")
            
        f_const = 1000
        # for f_const in self.npt_scheme:
        if not os.path.exists(f"{folder_path}/posre_nvt/posre_fc{f_const}.itp"):
            cmd = f"printf '2\n' | gmx genrestr -f {gro_filename}.gro -o posre_nvt/posre_fc{f_const}.itp -fc {f_const}"
            self._run_command(cmd, cwd=folder_path)

        shutil.copy(
            f"{folder_path}/posre_nvt/posre_fc{f_const}.itp", 
            f"{folder_path}/posre.itp"
        )
        
        # Generate nvt.tpr
        if not os.path.exists(f"{folder_path}/nvt{i}_{temp}K.tpr"):
            cmd = (f"gmx grompp -f nvt_{temp}K.mdp -c {gro_filename}.gro -r {gro_filename}.gro "
                   f"-p topol.top -o nvt{i}_{temp}K.tpr -maxwarn 1")
            self._run_command(cmd, cwd=folder_path)

        if not os.path.exists(f"{folder_path}/nvt{i}_{temp}K.gro"):
            # Run NVT
            cmd = f"gmx mdrun -v -deffnm nvt{i}_{temp}K -nt {self.n_threads} -nb gpu"
            print("Running NVT equilibration...")
            self._run_command(cmd, cwd=folder_path)
            self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_nvt")
            print("✓ NVT equilibration complete\n")

        self.plot(
            name_of_run="NVT", 
            tpr_name=f"nvt{i}_{temp}K", 
            edr_target_key=["Temperature", "Potential", "Kinetic-En.", "Pressure"], 
            folder_path=folder_path, 
            trj_type="trr", 
        )

        print(f"Delete posre_fc{f_const}.itp from the running directory\n")
        os.remove(f"{folder_path}/posre.itp")
        
        top_filename = f"{top_filename}_nvt{i}"
        gro_filename = f"nvt{i}_{temp}K"
        return gro_filename, top_filename
    
    def _run_npt_equilibration(self, folder_path, gro_filename, top_filename, temp):
        """
        Run NPT equilibration (constant N, P, T).
        
        Args:
            folder: Folder name
            temp: Temperature in Kelvin
        """
        print(f"\n{'='*60}")
        print(f"NPT EQUILIBRATION (T = {temp} K, P = 1 bar) for {folder_path}")
        print(f"{'='*60}\n")

        # Generate posres.itp
        if not os.path.exists(f"{folder_path}/posre"):
            os.makedirs(f"{folder_path}/posre", exist_ok=True)

        for f_const in self.npt_scheme:
            if not os.path.exists(f"{folder_path}/posre/posre_fc{f_const}.itp"):
                cmd = f"printf '2\n' | gmx genrestr -f {gro_filename}.gro -o posre/posre_fc{f_const}.itp -fc {f_const}"
                self._run_command(cmd, cwd=folder_path)
            
        if os.path.exists(f"{folder_path}/posre.itp"):
            os.remove(f"{folder_path}/posre.itp")

        # Create a list of descending fc list
        all_f_const = sorted(self.npt_scheme)
        all_f_const.reverse()
        # if i < self.em_nvt_iter:
        # all_f_const = all_f_const[:1]

        ## =============== Sequential NPT starts here ============
        top_filename = f"{top_filename}_npt"
        for f_const in all_f_const:
            # Copy the correct posre.itp
            print(f"Copy the posre_fc{f_const}.itp to the running directory")
            shutil.copy(
                f"{folder_path}/posre/posre_fc{f_const}.itp", 
                f"{folder_path}/posre.itp"
            )
            
            # Generate npt.tpr
            if not os.path.exists(f"{folder_path}/npt_{temp}K_fc{f_const}.tpr"):
                cmd = (f"gmx grompp -f npt_{temp}K_fc{f_const}.mdp -c {gro_filename}.gro "
                       f"-r {gro_filename}.gro -t {gro_filename}.cpt -p topol.top "
                       f"-o npt_{temp}K_fc{f_const}.tpr -maxwarn 2")
                self._run_command(cmd, cwd=folder_path)

            if not os.path.exists(f"{folder_path}/npt_{temp}K_fc{f_const}.gro"):
                # Run NPT
                cmd = f"gmx mdrun -v -deffnm npt_{temp}K_fc{f_const} -nt {self.n_threads} -nb gpu"
                print(f"***** Running NPT equilibration for fc = {f_const} ... *****")
                self._run_command(cmd, cwd=folder_path)
                self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_fc{f_const}")
                print(f"✓ NPT equilibration complete for fc = {f_const} \n")

            self.plot(
                name_of_run=f"NPT (fc={f_const})", 
                edr_target_key=["Temperature", "Potential", "Kinetic-En.", "Pressure"], 
                tpr_name=f"npt_{temp}K_fc{f_const}", 
                folder_path=folder_path, 
                trj_type="trr", 
            )
                
            top_filename = f"{top_filename}_fc{f_const}"
            gro_filename = f"npt_{temp}K_fc{f_const}"

            print(f"Delete posre_fc{f_const}.itp from the running directory\n")
            os.remove(f"{folder_path}/posre.itp")
            
        return gro_filename, top_filename

    def _run_relax_md(self, folder_path, gro_filename, top_filename, tot_time, temp, rep):
        print(f"\n{'='*60}")
        print(f"RELAXATION MD ({tot_time} ns) for {folder_path}, replicate {rep}")
        print(f"{'='*60}\n")
        
        tpr_name = f"md_relax_{temp}K_{temp}K_{tot_time}ns_r{rep}"

        # Fresh start
        if not os.path.exists(f"{folder_path}/{tpr_name}_pbc_center_compact.xtc") and \
        not os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            cmd = (f"gmx grompp -f md_relax_{temp}K_{temp}K_{tot_time}ns.mdp "
                   f"-c {gro_filename}.gro -t {gro_filename}.cpt "
                   f"-r {gro_filename}.gro -p topol.top -o {tpr_name}.tpr")
            self._run_command(cmd, cwd=folder_path)
            
            # Run production MD
            cmd = (f"gmx mdrun -gpu_id {self.gpu_id} -nt {self.n_threads} -reseed {rep*2+12345} "
                   f"-v -deffnm {tpr_name} -nb gpu")
            print(f"Running relaxation MD ...")
            self._run_command(cmd, cwd=folder_path)
            print(f"✓ Relaxation MD complete\n")

            # Ensure xtc protein do not cross the box boundary
            print(f"Center with pbc for relaxation MD...")
            cmd = f"printf 'Protein\nSystem\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}.xtc -o {tpr_name}_pbc_center_compact.xtc -pbc mol -center -ur compact" # 1 means just center the protein -> 0 means output the whole system
            self._run_command(cmd, cwd=folder_path)
    
        # If previous record exists
        elif os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            # Get the last checkpoint
            if os.path.exists(f"{folder_path}/{tpr_name}_prev.cpt"):
                if os.path.exists(f"{folder_path}/{tpr_name}.cpt"):
                    os.rename(
                        f"{folder_path}/{tpr_name}.cpt", 
                        f"{folder_path}/{tpr_name}_discard.cpt"
                    )
                os.rename(
                    f"{folder_path}/{tpr_name}_prev.cpt", 
                    f"{folder_path}/{tpr_name}.cpt"
                )
                if os.path.exists(f"{folder_path}/{tpr_name}_discard.cpt"):
                    os.remove(f"{folder_path}/{tpr_name}_discard.cpt")
                    
            # Run production MD
            cmd = (f"gmx mdrun -gpu_id {self.gpu_id} -nt {self.n_threads} -reseed {rep*2+12345} "
                   f"-v -deffnm {tpr_name} -nb gpu -cpi {tpr_name}.cpt -noappend")
            print(f"Running relaxation MD ...")
            self._run_command(cmd, cwd=folder_path)
            print(f"✓ Relaxation MD complete\n")

            # self.concat_trj(
            #     folder_path=folder_path, 
            #     tpr_name=tpr_name
            # )
            
            # Ensure xtc protein do not cross the box boundary
            # print(f"Center with pbc for relaxation MD...")
            # cmd = f"printf 'Protein\nSystem\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}.xtc -o {tpr_name}_pbc_center_compact.xtc -pbc mol -center -ur compact" # 1 means just center the protein -> 0 means output the whole system
            # self._run_command(cmd, cwd=folder_path)

            
        
        if self.delete_old_xtc and os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            os.remove(f"{folder_path}/{tpr_name}.xtc")
            
        self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_mdrelax")
        print(f"xtc corrected according to the periodic boundary conditions for relax MD ...")
        # quit()

        self.plot(
            name_of_run=f"relxation", 
            tpr_name=tpr_name, 
            folder_path=folder_path, 
        )

        gro_filename = f"{tpr_name}"
        top_filename = f"{top_filename}_mdrelax"
        return gro_filename, top_filename
    
        
    def _run_production_md(self, folder_path, gro_filename, top_filename, tot_time, temp_start, temp_end, rep):
        """
        Run production MD simulations with replicates.
        
        Args:
            folder: Folder name
            temp: Temperature in Kelvin
            tot_time: Total simulation time in nanoseconds
        """
        print(f"\n{'='*60}")
        print(f"PRODUCTION MD ({tot_time} ns) for {folder_path}, repliace {rep}")
        print(f"{'='*60}\n")

        # identify the min npt fc
        # min_fc = min(self.npt_scheme.keys())
        # assert f"npt_{temp_start}K_fc{min_fc}" == gro_filename
        
        # for i in range(self.replicate): # 
        # Generate md.tpr
        tpr_name = f"md_{temp_start}K_{temp_end}K_{tot_time}ns_r{rep}"

        # Fresh start
        if not os.path.exists(f"{folder_path}/{tpr_name}_pbc_center_compact.xtc") and \
        not os.path.exists(f"{tpr_name}_b{self.snapshot_begin_at_ps}.xtc") and \
        not os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            print(f"\n--- Replicate {rep} ---")
            if not os.path.exists(f"{folder_path}/{tpr_name}.tpr"):
                cmd = (f"gmx grompp -f md_{temp_start}K_{temp_end}K_{tot_time}ns.mdp "
                       f"-c {gro_filename}.gro -t {gro_filename}.cpt "
                       f"-r {gro_filename}.gro -p {top_filename}.top -o {tpr_name}.tpr")
                # gmx grompp -f md_310K_310K_100ns.mdp -c md_relax_310K_310K_30ns_r0.gro -t md_relax_310K_310K_30ns_r0.cpt -r md_relax_310K_310K_30ns_r0.gro -p topol.top -o md_310K_310K_100ns_r0.tpr
                self._run_command(cmd, cwd=folder_path)
            
            # Run production MD
            cmd = f"gmx mdrun -gpu_id {self.gpu_id} -nt {self.n_threads} -reseed {rep*2+12345} -v -deffnm {tpr_name} -nb gpu"
            if self.plumed != "":
                cmd += f" -plumed {self.plumed}"
            cmd = (cmd)
            print(f"Running production MD replicate {rep}...")
            self._run_command(cmd, cwd=folder_path)
            print(f"✓ Replicate {rep} complete\n")

            # trim xtc according to beginning
            print(f"Post-processing of xtc for replicate {rep}...")
            if self.snapshot_begin_at_ps > 0:
                print(f"xtc snapshot begins at {self.snapshot_begin_at_ps} ns replicate {rep}...")
                cmd = f"printf 'Protein\nSystem\n' | gmx trjconv -f {tpr_name}.xtc -o {tpr_name}_b{self.snapshot_begin_at_ps}.xtc -b {self.snapshot_begin_at_ps} -dt {self.dt}"
                self._run_command(cmd, cwd=folder_path)

                if self.delete_old_xtc:
                    os.remove(f"{folder_path}/{tpr_name}.xtc")
                    
                old_xtc_name = f"{tpr_name}_b{self.snapshot_begin_at_ps}"

            # Ensure xtc protein do not cross the box boundary
            print(f"Center with pbc for replicate {rep}...")
            cmd = f"printf 'Protein\nSystem\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}.xtc -o {tpr_name}_pbc_center_compact.xtc -pbc mol -center -ur compact" # 1 means just center the protein -> 0 means output the whole system
            self._run_command(cmd, cwd=folder_path)

        # If previous record exists
        elif os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            # Get the last checkpoint
            if os.path.exists(f"{folder_path}/{tpr_name}_prev.cpt"):
                if os.path.exists(f"{folder_path}/{tpr_name}.cpt"):
                    os.rename(
                        f"{folder_path}/{tpr_name}.cpt", 
                        f"{folder_path}/{tpr_name}_discard.cpt"
                    )
                os.rename(
                    f"{folder_path}/{tpr_name}_prev.cpt", 
                    f"{folder_path}/{tpr_name}.cpt"
                )
                if os.path.exists(f"{folder_path}/{tpr_name}_discard.cpt"):
                    os.remove(f"{folder_path}/{tpr_name}_discard.cpt")
                    
            # Run production MD
            cmd = (f"gmx mdrun -gpu_id {self.gpu_id} -nt {self.n_threads} -reseed {rep*2+12345} "
                   f"-v -deffnm {tpr_name} -nb gpu -cpi {tpr_name}.cpt -noappend")
            print(f"Running production MD replicate {rep}...")
            self._run_command(cmd, cwd=folder_path)
            print(f"✓ Replicate {rep} complete\n")

            # self.concat_trj(
            #     folder_path=folder_path, 
            #     tpr_name=tpr_name
            # )
            
            # Ensure xtc protein do not cross the box boundary
            # print(f"Center with pbc for replicate {rep}...")
            # cmd = f"printf 'Protein\nSystem\n' | gmx trjconv -s {tpr_name}.tpr -f {tpr_name}.xtc -o {tpr_name}_pbc_center_compact.xtc -pbc mol -center -ur compact" # 1 means just center the protein -> 0 means output the whole system
            # self._run_command(cmd, cwd=folder_path)
            
        if self.delete_old_xtc and os.path.exists(f"{folder_path}/{tpr_name}.xtc"):
            os.remove(f"{folder_path}/{tpr_name}.xtc")
        
        # Save snapshot(s) as pdb
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self.save_snapshot, folder_path, tpr_name, idx) for idx in self.md_snapshot_idx]
            for fut in futures:
                fut.result()
            
        self._rename_and_remove_topol(folder_path=folder_path, new_topol_name=f"{top_filename}_md{rep}")
        print(f"xtc corrected according to the periodic boundary conditions for replicate {rep}...")

        if self.plumed == "":
            self.plot(
                name_of_run=f"production", 
                tpr_name=tpr_name, 
                folder_path=folder_path, 
            )
        
        self.compile_rmsf_csv(
            tpr_name=tpr_name, 
            folder_path=folder_path, 
        )
        return temp_start, temp_end

    def compile_rmsf_csv(self, tpr_name, folder_path):
        # Collect all rmsf into one csv
        if not os.path.exists(f"{folder_path}/{tpr_name}_rmsf.csv"):
            DF = pd.DataFrame()
            cols = []
            for i in range(self.replicate):
                if os.path.exists(f"{folder_path}/{tpr_name}_r{i}_rmsf.xvg"):
                    df = pd.read_csv(f"{folder_path}/{tpr_name}_r{i}_rmsf.xvg", 
                                     skiprows=17, 
                                     header=None, 
                                     names=["Residue", "RMSF"], 
                                     sep='  ')
                    DF[f"r{i}"] = df["RMSF"]
                    cols.append(f"r{i}")
            DF["avg"] = DF[cols].mean(axis=1)
            if self.replicate > 1:
                DF["std"] = DF[cols].std(axis=1)
            DF.to_csv(f"{folder_path}/{tpr_name}_rmsf.csv", index=False)
            
        print(f"{'='*60}")
        print(f"ALL {self.replicate} REPLICATES COMPLETE for {folder_path}")
        print(f"{'='*60}\n")

            
    def run_single(self, idx, row):
        print("\n" + "#"*80)
        folder = row['folder']
        file = f"{row['file']}.pdb"
        os.makedirs(f"{self.destination_dir}/{folder}", exist_ok=True)

        print(f"# CLIENT {idx + 1}/{len(self.client_list)}: {folder}")
        temp_end = None
        if self.task in ("md", "relax_md"):
            temp_start = row['temp_start']
            temp_end = row['temp_end']
            tot_time = self.md_tot_time_ns # row['tot_time']
        else:
            temp_start = None
            tot_time = None
        
        print(f"# Temperature: {temp_start} to {temp_end} K, Time: {tot_time} ns")
        # print(f"# add_H: {add_h}, md: {run_md}")
        print("#"*80 + "\n")
        
        destination_path = Path(self.destination_dir) / folder

        # Move pdb file from soure_dir to destination_dir
        if self.source_dir != self.destination_dir and not os.path.exists(f"{destination_path}/{file}"):
            os.rename(f"{self.source_dir}/{file}", f"{destination_path}/{file}")
            
        # Create GromacsScriptWriter instance for this client
        writer = GromacsScriptWriter(
            destination_dir=destination_path,
            dt=self.dt,
            dt_em=self.dt_em,
            em_max_steps=self.em_max_steps,
            snapshot_interval=self.snapshot_interval
        )
        
        # Generate MDP files
        writer.write_all_scripts(
            temp_start=temp_start, temp_end=temp_end, 
            relax_md_tot_time_ns=self.relax_md_tot_time_ns, 
            tot_time=tot_time, npt_scheme=self.npt_scheme, task=self.task
        )
        
        # Execute based on flags
        # if add_h == 1 and run_md == 0:
        #     Only add hydrogens
        #     self._add_hydrogen_only(folder_path, file, temp)
        
        # elif run_md == 1:
            # Full MD workflow

        
        gro_filename, top_filename = self._run_preparation_steps(destination_path, file)
        if self.task in ['solvate', 'solvate_with_ion']:
            return

        # EM - NVT cycle
        if self.run_from_start or self.task in ("em", "nvt"):
            for i in range(self.em_nvt_iter):
                gro_filename, top_filename = self._run_energy_minimization(destination_path, gro_filename, top_filename, i=i)
                if self.task == 'em':
                    return
                gro_filename, top_filename = self._run_nvt_equilibration(
                    folder_path=destination_path, 
                    gro_filename=gro_filename, 
                    top_filename=top_filename, 
                    temp=temp_start, 
                    i=i
                )

        if self.run_from_start or self.task == "npt":
            gro_filename, top_filename = self._run_npt_equilibration(
                folder_path=destination_path, 
                gro_filename=gro_filename, 
                top_filename=top_filename, 
                temp=temp_start, 
                # i = self.em_nvt_iter, 
            )
        
        for r in range(self.start_rep, self.replicate):
            if self.relax_md_tot_time_ns > 0:
                gro_filename, top_filename = self._run_relax_md(
                    folder_path=destination_path, 
                    gro_filename=gro_filename, 
                    top_filename=top_filename, 
                    tot_time=self.relax_md_tot_time_ns, 
                    temp=temp_start, 
                    rep=r, 
                )
            if self.task == 'relax_md':
                continue
            # gro_filename = "md_relax_310K_310K_30ns_r0"
            # top_filename = "topol_dodecahedron_1_tip4p_NA_CL_0.15_em0_nvt0_em1_nvt1_em2_nvt2_em3_nvt3_em4_nvt4_em5_nvt5_em6_nvt6_em7_nvt7_em8_nvt8_em9_nvt9_npt_fc1000_fc316_fc100_fc40_fc16_fc7_fc3_fc1.2_fc0.5_fc0.2_fc0" + "_mdrelax" # * (r+1)
            temp_start, temp_end = self._run_production_md(
                folder_path=destination_path, 
                gro_filename=gro_filename, 
                top_filename=top_filename, 
                tot_time=tot_time, 
                temp_start=temp_start, 
                temp_end=temp_end, 
                rep=r, 
            )
        
        # else:
        #     print("⚠ No action requested (both add_H and md are 0)")

        # Move pdb back to source_dir --> tb done at run_gmx.py with parallel
        # if self.return_pdb_to_source:
        #     os.rename(f"{destination_path}/{file}", f"{self.source_dir}/{file}")
            
        print(f"\n✓ CLIENT {idx + 1} COMPLETE: {folder}\n")
    
    def run(self):
        """
        Execute all GROMACS simulations according to the client_list DataFrame.
        """
        print("\n" + "="*80)
        print("GROMACS RUNNER - STARTING BATCH SIMULATIONS")
        print("="*80 + "\n")

        # Ensure directory
        for folder in ["source_dir", "destination_dir"]:
            os.makedirs(getattr(self, folder), exist_ok=True)
        
        if self.allow_parallel_cpu_for_multiple_sample and self.task not in ["em", "md"]:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self.run_single, idx, row) for idx, row in self.client_list.iterrows()]
                for fut in futures:
                    fut.result()
        else:
            for idx, row in self.client_list.iterrows():
                try:
                    self.run_single(idx=idx, row=row)
                except Exception as e:
                    with open(f"error.txt", "a+") as out:
                        out.write(f"Index: {idx}\n")
                        out.write(f"{row}\n")
                        out.write(f"{e}\n")
                        traceback.print_exc(file=out)  # 
                        out.write(f"\n***************\n\n")
        
        print("\n" + "="*80)
        print("ALL SIMULATIONS COMPLETE!")
        print("="*80 + "\n")

