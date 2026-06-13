# Installs the Julia package set the planner needs.
#   Usage:  julia planner/install.jl     (or, from planner/:  julia install.jl)
# CUDA is optional — the planner falls back to CPU if no GPU is present.
using Pkg

Pkg.add([
    "POMDPs",
    "POMCPOW",
    "POMDPTools",
    "BasicPOMCP",
    "ParticleFilters",
    "Flux",
    "CUDA",
    "BSON",
    "JLD2",
    "JSON",
    "JSON3",
    "HTTP",
    "Distributions",
    "StatsBase",
])

# Statistics, Random, LinearAlgebra are part of the Julia standard library.
@info "Planner dependencies installed."
