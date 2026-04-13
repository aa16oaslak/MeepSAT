---
# 🛠️ Installation
---

[TOC]

Due to the complex dependencies of the MEEP FDTD software, it is highly recommended to install this package using the **Conda** package manager. More details are mentioned here in the official documentation of MEEP: https://meep.readthedocs.io/en/latest/Installation/

`Note`: We will be soon editing some segments of the MEEP base code to implement effective material approximation approach in our simulations. After that, MeepSAT will have its own version of MEEP. But for now, you can move forward with the mentioned installation guidelines.


Building using conda and git repository (recommended)
--------------------

**Create a new Conda environment and Install MEEP using Conda:**
    
pymeep:

```bash
conda create -n meep -c conda-forge pymeep
```

If you want to install parallel version of pymeep (**recommended**):

```bash
conda create -n parallel_meep -c conda-forge pymeep=*=mpi_mpich_*
```

**Activate and check if the conda environment is installed properly or not:**
    
pymeep:

```bash
conda activate meep
```

Install parallel pymeep and other dependencies:

```bash
conda activate parallel_meep
```

Check whether everything is working or not:

```bash
python -c 'import meep'
```


**Install this project from the repository:**

For development installation (editable and **recommended** for collaborators to use this):

```bash
git clone https://github.com/aa16oaslak/MeepSAT.git
cd MeepSAT
pip install -e .
```

For public repository (once the project is public):

```bash
pip install git+https://github.com/aa16oaslak/MeepSAT.git
```

For private repository (requires authentication):

```bash
pip install git+https://'your_token'@github.com/aa16oaslak/MeepSAT.git
```

Replace `'your_token'` with your PAT token generated in your github account.
For collaborators: If you need a token for the above, please contact the author @ aak25@hi.is


**Check if everything is working with MeepSAT import or not**

```bash
python -c 'import meep as mp; import MeepSAT as mpsat; print("Yayy!")'
```

**For updating things to the latest version of MeepSAT, just use git pull**

```bash
cd /path/to/my-private-MeepSAT-repo-head-directory
git pull
```

For Additional Checks, you can use the following piece of code to see if everything is working correctly within MeepSAT:

```bash
conda activate your_meep_conda_environment
python
```

Within python environment, copy paste the following lines of code

```python
import meepsat.field_analysis
import meepsat.simulation_2D
import meepsat.meep_goemetry

# Check what's available in each module
print("Functions in meepsat.field_analysis:")
print([name for name in dir(meepsat.field_analysis) if not name.startswith('_')])

print("\nFunctions in meepsat.simulation_2D:")
print([name for name in dir(meepsat.simulation_2D) if not name.startswith('_')])

print("\nFunctions in meepsat.meep_goemetry:")
print([name for name in dir(meepsat.meep_goemetry) if not name.startswith('_')])

```

If you are getting an output with all the functions in the various modules of MeepSAT, then you are all set to do some FDTD sims using MeepSAT!

Building a MeepSAT container using Singularity/Apptainer
--------------------

Containers provide a reproducible and portable environment for running MeepSAT simulations. Using Singularity/Apptainer offers several key benefits:

- **Consistency**: Ensure consistent results across different machines, HPC environments and computational environments by packaging all dependencies, libraries, and configurations in a single image.
- **Portability**: Run MeepSAT seamlessly on different HPC clusters, cloud platforms, or local machines without worrying about environment compatibility issues. For e.g., if you want to install the single-float precision version of MEEP, then having a container really helps.
- **Ease of Sharing**: Distribute a single container image to collaborators instead of writing complex installation instructions, reducing setup time and potential errors. 

Before proceeding further, make sure that Singularity/Apptainer is installed on your system. For installation instructions, visit [Apptainer Installation Guide](https://apptainer.org/docs/admin/main/installation.html).

**Building the MeepSAT container:**

You can use following recipe DEF file for creating the SIF file

```bash
# Usage:

# Header

Bootstrap: docker
From: ubuntu:24.04

# Sections

################################################################################
%help
################################################################################

    Meep container based on Ubuntu 24.04 and the intructions from:

       https://meep.readthedocs.io/en/master/Build_From_Source/#ubuntu-1604-and-1804

    To build the container, execute on a local machine:

       sudo singularity build meep_single_precision.sif meep_single_precision.def

    Note:

    pip3 installations will go to the python virtual environment /meep/venv;
    hence, when using the container one needs to activate the python environment
    by sourcing /meep/env/bin/activate.

################################################################################
%post
################################################################################

    ln -snf /usr/share/zoneinfo/Europe/Stockholm /etc/localtime
    echo Europe/Stockholm > /etc/timezone

    export RPATH_FLAGS="-Wl,-rpath,/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi"
    export MY_LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi ${RPATH_FLAGS}"
    export MY_CPPFLAGS="-I/usr/local/include -I/usr/include/hdf5/openmpi"
    export PYTHON_EXECUTABLE=/meep/venv/bin/python3

    apt-get -y update

    apt-get -y install     \
        build-essential         \
        gfortran                \
        libblas-dev             \
        liblapack-dev           \
        libgmp-dev              \
        swig                    \
        libgsl-dev              \
        autoconf                \
        pkg-config              \
        libpng-dev              \
        git                     \
        guile-2.2-dev           \
        libfftw3-dev            \
        libhdf5-openmpi-dev     \
        hdf5-tools              \
        libpython3-dev          \
        python3-pip             \
        cmake                   \
        wget                    \
        libpmix-dev             \
        libslurm-dev            \

    mkdir -p /meep/install

    # Build OpenMPI with SLURM/PMI support
    cd /meep/install
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    tar -xzf openmpi-4.1.6.tar.gz
    cd openmpi-4.1.6
    ./configure --prefix=/usr/local \
                --with-pmix \
                --with-slurm \
                --enable-mpi-thread-multiple \
                --enable-shared
    make -j$(nproc) && make install
    ldconfig

    # Update environment for newly built OpenMPI
    export PATH=/usr/local/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

    cd /meep/install
    git clone https://github.com/NanoComp/harminv.git
    cd harminv/
    sh autogen.sh --enable-shared
    make && make install

    cd /meep/install
    git clone https://github.com/NanoComp/libctl.git
    cd libctl/
    sh autogen.sh --enable-shared
    make && make install

    cd /meep/install
    git clone https://github.com/NanoComp/h5utils.git
    cd h5utils/
    sh autogen.sh CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
    make && make install

    cd /meep/install
    git clone https://github.com/NanoComp/mpb.git
    cd mpb/
    sh autogen.sh --enable-shared CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}" --with-hermitian-eps
    make && make install

    cd /meep/install
    git clone https://github.com/HomerReid/libGDSII.git
    cd libGDSII/
    sh autogen.sh
    make && make install

    #################################################################
    # pip3 installations should go to meep python virtual environment
    #################################################################
    mkdir -p /meep/venv

    apt-get install -y python3-venv

    python3 -m venv /meep/venv
    . /meep/venv/bin/activate

    # Rebuild mpi4py with the new OpenMPI
    pip3 install --no-cache-dir --force-reinstall mpi4py
    pip3 install Cython==0.29.16

    export HDF5_MPI="ON"
    pip3 install --no-binary=h5py h5py

    pip3 install autograd
    pip3 install scipy
    pip3 install matplotlib>3.0.0
    pip3 install ffmpeg
    pip3 install psutil
    pip3 install pandas

    ############################################################################

    cd /meep/install
    git clone https://github.com/stevengj/nlopt.git
    cd nlopt/
    cmake -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" && make && make install

    cd /meep/install
    git clone https://github.com/NanoComp/meep.git
    cd meep/
    sh autogen.sh --enable-shared --with-mpi --with-openmp --enable-single PYTHON="${PYTHON_EXECUTABLE}" LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
    make && make install

    # Install MeepSAT - Clone to a temporary location
    cd /meep/install
    git clone https://github.com/aa16oaslak/MeepSAT.git
    # NOTE: IF YOU HAVE A TOKEN USE THAT IN THE ABOVE LINE 
    cd MeepSAT/
    pip install .  # Remove the -e flag for non-editable install


    ############################################################################
    # TODO: Add here the part which will build your program and/or scripts
    # that will be executed inside the container.
    ############################################################################

    ############################################################################
    # Clean-up (purge the temporary files which are used to install meep)
    ############################################################################

    rm -rf /meep/install

################################################################################
%environment
################################################################################

    export LC_ALL=C

    # Add the following lines so thaat Python can always find the meep (and nlopt) package:
    export PYTHONPATH=/usr/local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages

    # Set PATH to include virtual environment and custom OpenMPI
    export PATH=/usr/local/bin:/meep/venv/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
################################################################################
```

To build a container from the MeepSAT definition file, use:

```bash
apptainer build meepsat.sif meepsat.def
```

or if using older Singularity syntax:

```bash
singularity build meepsat.sif meepsat.def
```

The above command creates a `meepsat.sif` file containing the complete MeepSAT environment with all dependencies. 

**Editing/Updating the sif file**

There are mainly two ways of how you can update/edit your SIF file:

**Method 1: Rebuild from the DEF file (Recommended)**

This is the cleanest and most reproducible approach. Simply modify your `.def` file with the changes you need, then rebuild using the earlier mentioned commands.

**Method 2: Using sandbox for modifications (Quick Fix OR debugging)**

You can convert your SIF to a writable sandbox directory and make changes:

```bash
apptainer build --sandbox meepsat_sandbox/ meepsat.sif
```

This creates a writable directory structure instead of a read-only SIF file. You can enter the sandbox interactively and edit the segments of your code:

```bash
apptainer shell --writable meepsat_sandbox/
```

Inside the sandbox, you can:
- Install new packages: `pip install your_package`
- Modify configuration files
- Test changes interactively
- Debug issues

Once you're satisfied with your changes, convert it back to a SIF file:

```bash
apptainer build meepsat_updated.sif meepsat_sandbox/
```