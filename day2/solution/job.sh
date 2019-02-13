#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:15:00 
#SBATCH --output job.out
#SBATCH --error job.err
#SBATCH --partition skl_usr_dbg
#SBATCH --account=train_cINFN19
#SBATCH -D .

ulimit -s unlimited

source /marconi/home/usertrain/a08tra37/stimulate/armadillo.env

make clean
make 

mpirun -np 48 ./model 
