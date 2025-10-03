#!/bin/bash

############ ============ IF YOU WANT TO RUN YOUR SIMS IN PARALLEL MEEP USING SBATCH IN CLUSTERS #############
# #SBATCH --job-name=simple_single_lens # Job name
# #SBATCH --partition=jon          # Partition name
# #SBATCH --nodes=1                # Number of nodes
# #SBATCH --ntasks=48               # Total number of MPI processes (4 per node × 2 nodes)
# #SBATCH --ntasks-per-node=48     # MPI processes per node
# #SBATCH --cpus-per-task=1        # CPU cores per MPI process
# #SBATCH --mem=0                  # Request all available memory per node
# #SBATCH --time=100:00:00          # Time limit hrs:min:sec
# #SBATCH --output=../output_files/job.log  # Standard output and error log

# For solving the issue with HDF5 file locking
# This is necessary for running MEEP simulations in parallel in SUNRISE
# export HDF5_USE_FILE_LOCKING=FALSE

# # Activating the conda parallel MEEP environment
# source ~/.bashrc
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate /cfs/home/asab1238/miniconda/installed_path/envs/parallel_meep

# # #~==============================================================================
# python_script_file_name=simple_single_lens

# # file name
# file_name=${python_script_file_name}

# # output directory + file
# base_output_dir=../output_files
# mkdir -p "$base_output_dir"


# # JSON file path
# json_file_path=/cfs/data/asab1238/MEEPSAT_WFH/Tutorials/taurus_3_lens_sims/sunrise_sims/taurus_f2f_meeting_2025/LMF/multi_freq/gaussianwaves/simple_single_lens/sim_files/simple_single_lens.json
# #~ Uncomment to set the frequency range
# # freq_start=0.30  # ~90 GHz
# # freq_end=0.60    # ~150 GHz  
# # freq_step=0.02   #~ 6 GHz step

# # Define resolution range
# resolutions=(10) #20)  # Add your desired resolutions here
# runtime=400 # ~Runtime in MEEP units

# # Generate a list of frequencies to simulate
# freqs=(0.30 0.40 0.50)

# # Beam waist
# beam_waist=(1.9434239 1.45756792 1.16605434) # ~ in mm for 90, 120, 150 GHz respectively
# #beam_waist=(6 4.5 3.6) # ~ in mm for 90, 120, 150 GHz respectively
# #~ Uncomment to set the frequency range
# # freq=$freq_start
# # while (( $(echo "$freq <= $freq_end" | bc -l) )); do
# #     freqs+=("$freq")
# #     freq=$(echo "$freq + $freq_step" | bc -l)
# # done

# # Create a temporary file to store the simulation commands
# command_file=$(mktemp)

# # Generate simulation commands for each resolution and frequency combination
# for res in "${resolutions[@]}"; do
#     for i in "${!freqs[@]}"; do
#         freq=${freqs[$i]}
#         waist=${beam_waist[$i]}
#         real_freq_ghz=$(echo "scale=6; $freq * 300" | bc -l)
#         freq_dir_name=$(printf "freq_%.1fGHz" $real_freq_ghz)
        
#         output_dir="${base_output_dir}/${res}/${freq_dir_name}/"
#         mkdir -p "$output_dir"
        
#         echo "mpirun -np 8 python $file_name.py $json_file_path $freq $res $output_dir $runtime $waist > $output_dir/$file_name.out 2> $output_dir/$file_name.err" >> $command_file
#     done
# done

# # Run 4 simulations in parallel (adjust -j to control the number of concurrent jobs)
# parallel -j 6 < $command_file

# # Clean up
# rm $command_file
# # #~==============================================================================
#~ For MEEPSAT simulations ON my local machine
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate meep_single_test

python_script_file_name=simple_single_lens

# file name
file_name=${python_script_file_name}

# output directory + file
base_output_dir=../output_files
mkdir -p "$base_output_dir"

# JSON file path
json_file_path=/home/aak/Phd_work/MEEPSAT_WFH/examples/python_scripts/simple_single_lens/sim_files/simple_single_lens.json

# Define resolution range
resolutions=(12) #20)  # Add your desired resolutions here
runtime=400 # ~Runtime in MEEP units

# Generate a list of frequencies to simulate
freqs=(0.30 0.40 0.50) 

# Beam waist
beam_waist=(1.9434239 1.45756792 1.16605434) # ~ in mm for 90, 120, 150 GHz respectively

# Generate simulation commands for each resolution and frequency combination
for res in "${resolutions[@]}"; do
    for i in "${!freqs[@]}"; do
        freq=${freqs[$i]}
        waist=${beam_waist[$i]}
        real_freq_ghz=$(echo "scale=6; $freq * 300" | bc -l)
        freq_dir_name=$(printf "freq_%.1fGHz" $real_freq_ghz)
        
        output_dir="${base_output_dir}/${res}/${freq_dir_name}/"
        mkdir -p "$output_dir"
        python $file_name.py $json_file_path $freq $res $output_dir $runtime $waist > $output_dir/$file_name.out 2> $output_dir/$file_name.err
    done
done

