This directory contains the instructions on how to run MeepSAT simulations in HPC environments using bash scripts

The idea is to have a directory like this:

```
project_directory
    ├── sim_files/
        ├── project_name.py 
        ├── project_name.sh
        ├── project_name.json 
    ├── output_files/
```

Then you can play with the bash script and python file to run simulations in HPC using `mpirun` OR `srun`.