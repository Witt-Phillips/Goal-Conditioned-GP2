include("bayesianGP.jl")
using Gen

function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    tr, = generate(model, (xs,), choicemap(:ys => ys))
    return tr
end

@gen function random_node_path(n::Kernel)
    if ({:stop} ~ bernoulli(isa(n, PrimitiveKernel) ? 1.0 : 0.5))
        return :tree
    else
        (next_node, direction) = ({:left} ~ bernoulli(0.5)) ? (n.left, :left) : (n.right, :right)
        rest_of_path ~ random_node_path(next_node)
        
        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end
        
    end
end;

@gen function regen_random_subtree_randomness(prev_trace)
    path ~ random_node_path(get_retval(prev_trace))
    new_subtree ~ covariance_prior()
    return path
end;

function subtree_involution(trace, forward_choices, path_to_subtree, proposal_args)
    # Need to return a new trace, backward_choices, and a weight.
    backward_choices = choicemap()
    
    # In the backward direction, the `random_node_path` function should
    # make all the same choices, so that the same exact node is reached
    # for resimulation.
    set_submap!(backward_choices, :path, get_submap(forward_choices, :path))
    
    # But in the backward direction, the `:new_subtree` generation should
    # produce the *existing* subtree.
    set_submap!(backward_choices, :new_subtree, get_submap(get_choices(trace), path_to_subtree))
    
    # The new trace should be just like the old one, but we are updating everything
    # about the new subtree.
    new_trace_choices = choicemap()
    set_submap!(new_trace_choices, path_to_subtree, get_submap(forward_choices, :new_subtree))
    
    # Run update and get the new weight.
    new_trace, weight, = update(trace, get_args(trace), (UnknownChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end