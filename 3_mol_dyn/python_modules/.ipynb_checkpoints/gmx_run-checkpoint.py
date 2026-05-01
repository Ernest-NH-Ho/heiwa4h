from python_modules.gmx_run_config import Gmx_Scripts, Gmx_Config

class ScriptWriter:
    def __init__(self, destination_dir, dt=0.001, dt_em=0.01, em_max_steps=50000, snapshot_interval=5000):
        self.destination_dir = destination_dir
        self.dt = dt
        self.dt_em = dt_em
        self.snapshot_interval = snapshot_interval
        self.em_max_steps = em_max_steps
        
    def _write_ions_mdp(self, tot_time):
        with open(f"{self.destination_dir}/ion.mdp", "a+") as file:
            file.write(f"; ions.mdp - used as input into grompp to generate ions.tpr\n")
            file.write(f"; Parameters describing what to do, when to stop and what to save\n")
            file.write(f"integrator  = steep         ; Algorithm (steep = steepest descent minimization)\n")
            file.write(f"emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm\n")
            file.write(f"emstep      = 0.01          ; Minimization step size\n")
            file.write(f"nsteps      = {tot_time//(self.dt*2)}       ; Maximum number of (minimization) steps to perform\n")
            file.write(f"\n")
            file.write(f"; Parameters describing how to find the neighbors of each atom and how to calculate the interactions\n")
            file.write(f"nstlist         = 1         ; Frequency to update the neighbor list and long range forces\n")
            file.write(f"cutoff-scheme	= Verlet    ; Buffered neighbor searching \n")
            file.write(f"ns_type         = grid      ; Method to determine neighbor list (simple, grid)\n")
            file.write(f"coulombtype     = cutoff    ; Treatment of long range electrostatic interactions\n")
            file.write(f"rcoulomb        = 1.0       ; Short-range electrostatic cut-off\n")
            file.write(f"rvdw            = 1.0       ; Short-range Van der Waals cut-off\n")
            file.write(f"pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions\n")
    def _write_md_mdp(self, temp, tot_time):
        with open(f"{self.destination_dir}/md_{temp}K_{tot_time}ns.mdp", "a+") as file:
            file.write(f"title                   = HMlCA NPT Production\n")
            file.write(f"; Run parameters\n")
            file.write(f"integrator              = md        ; leap-frog integrator\n")
            file.write(f"nsteps                  = {tot_time//self.dt}  ; 2 * 10000000 = 20000 ps (20 ns)\n")
            file.write(f"dt                      = {self.dt*2}     ; 2 fs\n")
            file.write(f"; Output control\n")
            file.write(f"nstxout                 = 0         ; suppress bulky .trr file by specifying \n")
            file.write(f"nstvout                 = 0         ; 0 for output frequency of nstxout,\n")
            file.write(f"nstfout                 = 0         ; nstvout, and nstfout\n")
            file.write(f"nstenergy               = {self.snapshot_interval}      ; save energies every 10.0 ps\n")
            file.write(f"nstlog                  = {self.snapshot_interval}      ; update log file every 10.0 ps\n")
            file.write(f"nstxout-compressed      = {self.snapshot_interval}      ; save compressed coordinates every 10.0 ps\n")
            file.write(f"compressed-x-grps       = System    ; save the whole system\n")
            file.write(f"; Bond parameters\n")
            file.write(f"continuation            = yes       ; Restarting after NPT \n")
            file.write(f"constraint_algorithm    = lincs     ; holonomic constraints \n")
            file.write(f"constraints             = h-bonds   ; bonds involving H are constrained\n")
            file.write(f"lincs_iter              = 1         ; accuracy of LINCS\n")
            file.write(f"lincs_order             = 4         ; also related to accuracy\n")
            file.write(f"; Neighborsearching\n")
            file.write(f"cutoff-scheme           = Verlet    ; Buffered neighbor searching\n")
            file.write(f"ns_type                 = grid      ; search neighboring grid cells\n")
            file.write(f"nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet scheme\n")
            file.write(f"rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)\n")
            file.write(f"rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)\n")
            file.write(f"; Electrostatics\n")
            file.write(f"coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics\n")
            file.write(f"pme_order               = 4         ; cubic interpolation\n")
            file.write(f"fourierspacing          = 0.16      ; grid spacing for FFT\n")
            file.write(f"; Temperature coupling is on\n")
            file.write(f"tcoupl                  = V-rescale             ; modified Berendsen thermostat\n")
            file.write(f"tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate\n")
            file.write(f"tau_t                   = 0.1     0.1           ; time constant, in ps\n")
            file.write(f"ref_t                   = {temp}     {temp}           ; reference temperature, one for each group, in K\n")
            file.write(f"; Pressure coupling is on\n")
            file.write(f"pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT\n")
            file.write(f"pcoupltype              = isotropic             ; uniform scaling of box vectors\n")
            file.write(f"tau_p                   = 2.0                   ; time constant, in ps\n")
            file.write(f"ref_p                   = 1.0                   ; reference pressure, in bar\n")
            file.write(f"compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1\n")
            file.write(f"; Periodic boundary conditions\n")
            file.write(f"pbc                     = xyz       ; 3-D PBC\n")
            file.write(f"; Dispersion correction\n")
            file.write(f"DispCorr                = EnerPres  ; account for cut-off vdW scheme\n")
            file.write(f"; Velocity generation\n")
            file.write(f"gen_vel                 = no        ; Velocity generation is off \n")

    


class Gmx_Runner:
    
    def __init__(self, config):
        """ Data source: 1. list of pdb name, 2. All folders in a parent dir """
        """ Objective: 1. Add H, 2. run MD  """
        self.config = config
        self.ori_dir_of_gmx_scripts = "/home/ncku-che/ErnestHo-NCKU/Gmx"
        self.gmx_scripts_files = ["ions.mdp", "md_310.15K_20ns.mdp"]
    
    
    def run(self, objective):
        