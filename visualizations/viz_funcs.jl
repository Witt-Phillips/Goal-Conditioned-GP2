using Plots

include("../src/inference/involutive_mcmc.jl")

function run_mcmc_viz(trace, frames::Int, iters_per_frame::Int)
    
    viz = @animate for frame=1:frames
        for iter in iters_per_frame
            trace, = mh(trace, regen_random_subtree_randomness, (), subtree_involution)
            trace, = mh(trace, select(:noise))
        end
        visualize_gp_trace(trace, minimum(xs), maximum(xs); title="Iter $(frame*iters_per_frame)/$(frames*iters_per_frame)")
    end
    
    println("Score: $(get_score(trace))")
    println("Final program:")
    println(get_retval(trace))
    gif(viz)
end

