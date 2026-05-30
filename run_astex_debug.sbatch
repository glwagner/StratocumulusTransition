#!/bin/bash
#SBATCH -A m5176_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -J astex_dev
#SBATCH -o astex-dev-%j.out
#SBATCH -e astex-dev-%j.err

# Quick smoke test of the ASTEX setup on the Perlmutter GPU *debug* queue
# (fast turnaround, 30-minute limit). Runs the coarse `dev` configuration:
#
#   sbatch run_astex_debug.sbatch
#
# Use this to confirm the model builds, stays stable (no NaNs), forms a cloud
# layer, and produces cloud-top radiative cooling before launching the full
# production job (run_astex.sbatch).

set -euo pipefail

export JULIA=/pscratch/sd/g/glwagner/.julia/juliaup/julia-1.12.4+0.x64.linux.gnu/bin/julia
export JULIA_DEPOT_PATH=/pscratch/sd/g/glwagner/.julia
export ASTEX_CONFIG=dev
export JULIA_NUM_THREADS=8
export JULIA_CUDA_MEMORY_POOL=none

cd /global/u1/g/glwagner/StratocumulusTransition

srun --ntasks=1 --gpus-per-task=1 $JULIA --project=. astex.jl
