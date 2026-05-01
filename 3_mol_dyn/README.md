## Purpose
Perform molecular dynamics simulation using Gromacs

## Install Gromacs 2026.0
1. Download
```wget https://ftp.gromacs.org/gromacs/gromacs-2026.0.tar.gz```

2. Installation with GPU support
```bash
tar xfz gromacs-2026.0.tar.gz
cd gromacs-2026.0
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_GPU=CUDA
make
make check
sudo make install
source /usr/local/gromacs/bin/GMXRC
```

## Prepare data
- Users are recommended to save one batch of pdb files within one folder (within or outside this directory are fine)
- The CSV file must contain the following fields:

| folder | file | temp_start | temp_end |
|--------|------|------------|----------|
| `folder` | Relative path to the folder where pdb files are saved |
| `file` | Name of individual pdb files excluding the suffix ".pdb" |
| `temp_start` | Starting temperature in Kelvin scale |
| `temp_end` | Final temperature in Kelvin scale |

## Prepare your script
For most users, all configurations can be edited at ```run_gmx.py``` in ```components```

## Run
```bash
python run_gmx.py -task md
```