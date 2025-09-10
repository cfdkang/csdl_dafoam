#!/bin/bash

cd ./folder_cfd

echo Delete optimization logs
rm -rf ASO_2DAF_*

echo Delete previous CFD cases
rm -rf ./openfoam_naca0012/{processor*, dRdWColoring_4*}

echo Run the script
mpirun -np 4 python mpi_tot.py
