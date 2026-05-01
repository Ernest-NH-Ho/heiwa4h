"""
GROMACS Script Writer Module
Generates MDP (Molecular Dynamics Parameter) files for GROMACS simulations.
"""

import os
from pathlib import Path


class GromacsScriptWriter:
    """Writes GROMACS MDP (molecular dynamics parameter) files."""
    
    def __init__(self, destination_dir, dt=0.002, dt_em=0.01, em_max_steps=50000, snapshot_interval=5000):
        """
        Initialize the GROMACS script writer.
        
        Args:
            destination_dir: Directory where PDB file is located and MDP files will be written
            dt: Timestep for MD simulations in ps (default: 0.002 ps = 2 fs)
            dt_em: Step size for energy minimization in nm (default: 0.01)
            em_max_steps: Maximum steps for energy minimization (default: 50000)
            snapshot_interval: Output frequency for production MD in steps (default: 5000)
        """
        self.destination_dir = Path(destination_dir)
        self.dt = dt
        self.dt_em = dt_em
        self.em_max_steps = em_max_steps
        self.snapshot_interval = snapshot_interval
        
        # Create destination directory if it doesn't exist
        self.destination_dir.mkdir(parents=True, exist_ok=True)
    
    def _write_ions_mdp(self):
        """
        Write ions.mdp file for pre-ion-addition energy minimization.
        This creates a template for genion to add counter ions.
        """
        filepath = self.destination_dir / "ions.mdp"
        
        content = f"""; ions.mdp - used as input into grompp to generate ions.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = {self.dt_em}          ; Minimization step size
nsteps      = {self.em_max_steps}         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbor list and long range forces
cutoff-scheme	= Verlet    ; Buffered neighbor searching 
ns_type         = grid      ; Method to determine neighbor list (simple, grid)
coulombtype     = cutoff    ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Written: {filepath}")
    
    def _write_minim_mdp(self):
        """
        Write minim.mdp file for energy minimization after ion addition.
        This is the real energy minimization step.
        """
        filepath = self.destination_dir / "minim.mdp"
        
        content = f"""; minim.mdp - used as input into grompp to generate em.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = {self.dt_em}          ; Minimization step size
nsteps      = {self.em_max_steps}         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbor searching
ns_type         = grid      ; Method to determine neighbor list (simple, grid)
coulombtype     = PME       ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Written: {filepath}")
    
    def _write_nvt_mdp(self, temp):
        """
        Write NVT equilibration file (constant N, V, T).
        Temperature equilibration with position restraints.
        
        Args:
            temp: Temperature in Kelvin (e.g., 310.15)
        """
        filepath = self.destination_dir / f"nvt_{temp}K.mdp"
        
        # NVT runs for 200 ps (50000 steps * 0.002 ps)
        nvt_steps = 50000
        
        content = f"""; nvt_{temp}K.mdp - NVT equilibration 
title                   = NVT equilibration 
define                  = -DPOSRES  ; position restrain the protein
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = {nvt_steps}     ; 2 * {nvt_steps} = {nvt_steps * self.dt} ps
dt                      = {self.dt}     ; {self.dt * 1000} fs
; Output control
nstxout                 = 500       ; save coordinates every 1.0 ps
nstvout                 = 500       ; save velocities every 1.0 ps
nstenergy               = 500       ; save energies every 1.0 ps
nstlog                  = 500       ; update log file every 1.0 ps
; Bond parameters
continuation            = no        ; first dynamics run
constraint_algorithm    = lincs     ; holonomic constraints 
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Nonbonded settings 
cutoff-scheme           = Verlet    ; Buffered neighbor searching
ns_type                 = grid      ; search neighboring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = {temp}     {temp}           ; reference temperature, one for each group, in K
; Pressure coupling is off
pcoupl                  = no        ; no pressure coupling in NVT
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Velocity generation
gen_vel                 = yes       ; assign velocities from Maxwell distribution
gen_temp                = {temp}       ; temperature for Maxwell distribution
gen_seed                = -1        ; generate a random seed
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Written: {filepath}")

    # def _write_posres_itp(self, npt_scheme):
        
    
    def _write_npt_mdp(self, temp, npt_scheme={1000:1000}):
        """
        Write NPT equilibration file (constant N, P, T).
        Pressure equilibration with position restraints.
        
        Args:
            temp: Temperature in Kelvin (e.g., 310.15)
        """
        for f_const in npt_scheme:
            filepath = self.destination_dir / f"npt_{temp}K_fc{f_const}.mdp"
            
            # NPT runs for 1000 ps = 1 ns (500000 steps * 0.002 ps)
            npt_steps = round(npt_scheme[f_const] / self.dt)
            
            content = f"""; npt_{temp}K.mdp - NPT equilibration 
    title                   = NPT equilibration 
    define                  = -DPOSRES  ; position restrain the protein
    ; Run parameters
    integrator              = md        ; leap-frog integrator
    nsteps                  = {npt_steps}     ; 2 * {npt_steps} = {npt_steps * self.dt} ps
    dt                      = {self.dt}     ; {self.dt * 1000} fs
    ; Output control
    nstxout                 = 500       ; save coordinates every 1.0 ps
    nstvout                 = 500       ; save velocities every 1.0 ps
    nstenergy               = 500       ; save energies every 1.0 ps
    nstlog                  = 500       ; update log file every 1.0 ps
    ; Bond parameters
    continuation            = yes       ; Restarting after NVT 
    constraint_algorithm    = lincs     ; holonomic constraints 
    constraints             = h-bonds   ; bonds involving H are constrained
    lincs_iter              = 1         ; accuracy of LINCS
    lincs_order             = 4         ; also related to accuracy
    ; Nonbonded settings 
    cutoff-scheme           = Verlet    ; Buffered neighbor searching
    ns_type                 = grid      ; search neighboring grid cells
    nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet scheme
    rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
    rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
    DispCorr                = EnerPres  ; account for cut-off vdW scheme
    ; Electrostatics
    coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
    pme_order               = 4         ; cubic interpolation
    fourierspacing          = 0.16      ; grid spacing for FFT
    ; Temperature coupling is on
    tcoupl                  = V-rescale             ; modified Berendsen thermostat
    tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
    tau_t                   = 0.1     0.1           ; time constant, in ps
    ref_t                   = {temp}     {temp}           ; reference temperature, one for each group, in K
    ; Pressure coupling is on
    pcoupl                  = Berendsen     ; Pressure coupling on in NPT *****
    pcoupltype              = isotropic             ; uniform scaling of box vectors
    tau_p                   = 1.0                   ; time constant, in ps
    ref_p                   = 1.0                   ; reference pressure, in bar
    compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1
    refcoord_scaling        = com
    ; Periodic boundary conditions
    pbc                     = xyz       ; 3-D PBC
    ; Velocity generation
    gen_vel                 = no        ; Velocity generation is off ** The whole purpose of previous equilibration is defeated if set to yes
    """
        
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✓ Written: {filepath}")
    
    def _write_md_mdp(self, tot_time, temp_start, temp_end, relax):
        """
        Write production MD file (no position restraints).
        
        Args:
            temp: Temperature in Kelvin (e.g., 310.15)
            tot_time: Total simulation time in nanoseconds (e.g., 20)
        """
        if relax:
            filepath = self.destination_dir / f"md_relax_{temp_start}K_{temp_end}K_{tot_time}ns.mdp"
        else:
            filepath = self.destination_dir / f"md_{temp_start}K_{temp_end}K_{tot_time}ns.mdp"
            
        # Calculate number of steps: tot_time (ns) * 1000 (ps/ns) / dt (ps/step)
        md_steps = int(tot_time * 1000 / self.dt)
        
        content = f"""; md_{temp_start}K_{temp_end}K_{tot_time}ns.mdp - NPT Production
title                   = NPT Production
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = {md_steps}  ; {self.dt} * {md_steps} = {md_steps * self.dt} ps ({tot_time} ns)
dt                      = {self.dt}     ; {self.dt * 1000} fs
; Output control
nstxout                 = 0         ; suppress bulky .trr file by specifying 
nstvout                 = 0         ; 0 for output frequency of nstxout,
nstfout                 = 0         ; nstvout, and nstfout
nstenergy               = {self.snapshot_interval}      ; save energies every {self.snapshot_interval * self.dt} ps
nstlog                  = {self.snapshot_interval}      ; update log file every {self.snapshot_interval * self.dt} ps
nstxout-compressed      = {self.snapshot_interval}      ; save compressed coordinates every {self.snapshot_interval * self.dt} ps
compressed-x-grps       = System    ; save the whole system
; Bond parameters -> degree of freedom of the system
continuation            = yes       ; Restarting after NPT 
constraint_algorithm    = lincs     ; holonomic constraints 
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Neighborsearching
cutoff-scheme           = Verlet    ; Buffered neighbor searching
ns_type                 = grid      ; search neighboring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet scheme
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat <-- Random Brownian motion
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = {temp_start}     {temp_start}           ; reference temperature, one for each group, in K
annealing           = single single   ;
annealing-npoints   = 2 2             ; there is only 1 temp_start + 1 temp_end
annealing-time          = 0 {1000*tot_time}  0 {1000*tot_time} ;ramping time in ps - (1000 is 1 ns)
annealing-temp          = {temp_start} {temp_end}   {temp_start} {temp_end}  ;start and stop temp
; Pressure coupling is on
pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT
pcoupltype              = isotropic             ; uniform scaling of box vectors
tau_p                   = 2.0                   ; time constant, in ps
ref_p                   = 1.0                   ; reference pressure, in bar
compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Dispersion correction
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Velocity generation
gen_vel                 = no        ; Velocity generation is off 
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Written: {filepath}")
    
    def write_all_scripts(self, tot_time, task, temp_start, relax_md_tot_time_ns, npt_scheme={1000:1000}, temp_end=None):
        """
        Write all MDP files needed for a complete GROMACS workflow.
        
        Args:
            temp: Temperature in Kelvin (e.g., 310.15)
            tot_time: Total production MD time in nanoseconds (e.g., 20)
        """
        print(f"\n{'='*60}")
        print(f"Writing MDP files to: {self.destination_dir}")
        print(f"Temperature: {temp_start} K to {temp_end} K, Production time: {tot_time} ns")
        print(f"{'='*60}\n")

        if task in ("solvate_with_ion", "em", "nvt", "npt", "md"):
            self._write_ions_mdp()
        if task in ("em", "nvt", "npt", "md"):
            self._write_minim_mdp()
        if task == "md":
            self._write_nvt_mdp(temp_start)
            # self._write_posres_itp(npt_scheme)
            self._write_npt_mdp(temp_start, npt_scheme=npt_scheme)
            if relax_md_tot_time_ns > 0:
                self._write_md_mdp(relax_md_tot_time_ns, temp_start, temp_start, relax=True)
            self._write_md_mdp(tot_time, temp_start, temp_end, relax=False)
        
        print(f"\n{'='*60}")
        print("All MDP files written successfully!")
        print(f"{'='*60}\n")
