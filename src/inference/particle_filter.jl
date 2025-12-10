#= Adapted from the AC codebase: https://github.com/CNCLgithub/AdaptiveComputation =#
using Gen

include("../model.jl")
include("involutive_mcmc.jl")
include("ac.jl")


"""
Particle Filter over Gaussian Process kernels, using Adaptive Computation.

Arguments:
- xs, ys: Training data
- num_particles: Number of particles
- num_rounds: Number of rejuvenation rounds  
- num_mcmc_moves: (deprecated, use budget instead)
- num_samples: Number of samples to return
- allocation: AllocationStrategy (UniformAllocation, TaskDrivenAC, etc.)
- budget: Total MCMC budget per round

Keyword arguments:
- verbose: Print allocation details
- return_history: Return allocation history for visualization
"""
function gp_particle_filter(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    num_particles::Int,
    num_rounds::Int,
    num_mcmc_moves::Int,
    num_samples::Int,
    allocation::AllocationStrategy = UniformAllocation(),
    budget::Int = 100;
    verbose=false,
    return_history=false
)
    obs = choicemap(:ys => ys)
    state = Gen.initialize_particle_filter(model, (xs,), obs, num_particles)
    
    # Track allocation history per round if requested
    allocation_history = return_history ? Vector{Vector{Int}}() : nothing
    
    for round in 1:num_rounds
        # Reset per-round tracking state (for strategies that track acceptance)
        reset_round!(allocation, num_particles)
        
        # Compute allocations BEFORE resampling 
        # (while scores/weights still differentiate particles)
        allocations = compute_allocations(
            allocation, 
            state.traces, 
            state.log_weights, 
            budget;
            verbose
        )
        
        # Record allocation history
        if return_history
            push!(allocation_history, copy(allocations))
        end
        
        # Now resample (this resets importance weights to uniform)
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)

        # Rejuvenate each particle according to its allocation
        for i in 1:num_particles
            for _ in 1:allocations[i]
                # Structure move
                state.traces[i], accepted = mh(
                    state.traces[i],
                    regen_random_subtree_randomness,
                    (),
                    subtree_involution
                )
                # Track acceptance for sensitivity estimation
                record_move!(allocation, i, accepted)
                
                # Noise hyperparameter move
                state.traces[i], = mh(state.traces[i], select(:noise))
            end
        end
    end
    
    samples = Gen.sample_unweighted_traces(state, num_samples)
    
    if return_history
        # Return final allocations per particle (sum across rounds)
        total_allocations = sum(allocation_history)
        return samples, allocation_history, total_allocations
    else
        return samples
    end
end
