#!/bin/bash

# Activating the conda MEEP environment
source activate meep

#SBATCH --job-name=aperture_stop

# file name
file_name=aperture_stop


# Run the Python script and redirect output
gdb --args python $file_name.py #> $file_name.out
# python $file_name.py > $file_name.out