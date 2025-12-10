using Gen

include("../model.jl")
include("involutive_mcmc.jl")
include("relevance.jl")


function gp_particle_filter(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    num_particles::Int,
    num_rounds::Int,
    num_mcmc_moves::Int,
    num_samples::Int,
    allocation::AllocationStrategy = UniformAllocation(),
    budget::Int = 100
)
    obs = choicemap(:ys => ys)
    state = Gen.initialize_particle_filter(model, (xs,), obs, num_particles)
    
    for round in 1:num_rounds
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        
        # Adaptive computation - determine the compute allocation for each particle
        allocations = compute_allocations(
            allocation, 
            state.traces, 
            state.log_weights, 
            budget
        )

        # rejuvenate
        for i in 1:num_particles
            # for _ in 1:num_mcmc_moves # AC GETS APPLIED HERE. EVERY PARTICLE GETS NUM_MCMC MOVES RIGHT NOW
            for _ in 1:allocations[i]
                state.traces[i], = mh(
                    state.traces[i],
                    regen_random_subtree_randomness,
                    (),
                    subtree_involution
                )

                # noise hyperparam
                state.traces[i], = mh(state.traces[i], select(:noise))
            end
        end
    end
    
    return Gen.sample_unweighted_traces(state, num_samples)
end