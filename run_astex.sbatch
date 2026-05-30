#!/bin/bash
#SBATCH -A m5176_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -J astex
#SBATCH -o astex-%j.out
#SBATCH -e astex-%j.err

# Perlmutter GPU batch job for the full-resolution (production) ASTEX
# stratocumulus-to-cumulus transition simulation (van der Dussen et al. 2013).
#
#   sbatch run_astex.sbatch
#
# The production config is the 4.48 × 4.48 km² domain at 35 m horizontal resolution
# on the EUCLIPSE stretched vertical grid (~426 levels, 5 m through the inversion),
# integrated for 40 hours with interactive RRTMGP radiation. Expect this to be long;
# adjust -t and consider checkpointing for very long runs.

set -euo pipefail

export JULIA=/pscratch/sd/g/glwagner/.julia/juliaup/julia-1.12.4+0.x64.linux.gnu/bin/julia
export JULIA_DEPOT_PATH=/pscratch/sd/g/glwagner/.julia
export ASTEX_CONFIG=production
export JULIA_NUM_THREADS=8
export JULIA_CUDA_MEMORY_POOL=none   # play nicely with Oceananigans/CUDA allocations

cd /global/u1/g/glwagner/StratocumulusTransition

srun --ntasks=1 --gpus-per-task=1 $JULIA --project=. astex.jl

echo "Production run complete. Analyze with:"
echo "  ASTEX_CONFIG=production $JULIA --project=. analyze_astex.jl"
