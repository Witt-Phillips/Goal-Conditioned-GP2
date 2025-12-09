#= Defines methods for determining the relative relevance of each particle, expressed as a weighted list =#

abstract type AllocationStrategy end

function compute_allocations(
    strategy::AllocationStrategy, 
    traces::Vector, 
    log_weights::Vector{Float64},
    total_budget::Int
)::Vector{Int}
    error("compute_allocations not implemented for $(typeof(strategy))")
end


#= Uniform Allocation - all particles get same amount of compute =#
struct UniformAllocation <: AllocationStrategy end

function compute_allocations(
    ::UniformAllocation, 
    traces::Vector, 
    log_weights::Vector{Float64},
    total_budget::Int
)
    n = length(traces)
    per_particle = total_budget ÷ n
    return fill(per_particle, n)
end