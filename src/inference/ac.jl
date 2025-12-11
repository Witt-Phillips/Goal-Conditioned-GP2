#= Adapted from the AC codebase: https://github.com/CNCLgithub/AdaptiveComputation =#

using Gen: logsumexp
using Statistics: mean

abstract type AllocationStrategy end

# Default implementations for strategies. don't track state
reset_round!(::AllocationStrategy, ::Int) = nothing
record_move!(::AllocationStrategy, ::Int, ::Bool) = nothing

function compute_allocations(
    strategy::AllocationStrategy,
    traces::Vector,
    log_weights::Vector{Float64},
    budget::Int;
    verbose=false
)::Vector{Int}
    error("compute_allocations not implemented for $(typeof(strategy))")
end

#= Uniform: stateless, equal allocation =#
struct UniformAllocation <: AllocationStrategy end

function compute_allocations(
    ::UniformAllocation,
    traces::Vector,
    log_weights::Vector{Float64},
    budget::Int;
    verbose=false
)
    n = length(traces)
    n == 0 && return Int[]
    per_particle = budget ÷ n
    return fill(per_particle, n)
end

#==============================================================================
  TASK-DRIVEN ADAPTIVE COMPUTATION
==============================================================================#

"""
Abstract type for task objectives.
Subtype this to define what "success" means for your inference task.
"""
abstract type TaskObjective end

"""
Prediction objective: measure predictive log-likelihood on held-out data.
"""
@kwdef struct PredictionObjective <: TaskObjective
    held_out_xs::Vector{Float64}
    held_out_ys::Vector{Float64}
end

"""
Compute the task objective value for a trace.
Higher = better performance on the task.
"""
function evaluate_objective(obj::PredictionObjective, trace)
    # Get the kernel and noise from the trace
    kernel = get_retval(trace)
    noise = trace[:noise]
    train_xs, = get_args(trace)
    train_ys = trace[:ys]
    
    # Predict on held-out data
    try
        pred_ys = predict_ys(kernel, noise, train_xs, train_ys, obj.held_out_xs)
        # Negative squared error (higher = better)
        mse = mean((pred_ys .- obj.held_out_ys).^2)
        return -mse
    catch
        return -Inf  # If prediction fails, worst score
    end
end

"""
Simple objective: just use the trace score (log-posterior).
Fallback when no task-specific objective is defined.
"""
struct PosteriorObjective <: TaskObjective end

function evaluate_objective(::PosteriorObjective, trace)
    return get_score(trace)
end


#==============================================================================
  EXTRAPOLATION
==============================================================================#

"""
Extrapolation objective with ground truth.

Measures prediction accuracy at points beyond the training range.

Parameters:
- extrap_xs: X values beyond training range
- extrap_ys: True Y values at those points
- direction: :right, :left, or :both (which side to extrapolate)
"""
@kwdef struct ExtrapolationObjective <: TaskObjective
    extrap_xs::Vector{Float64}
    extrap_ys::Vector{Float64}
end

function evaluate_objective(obj::ExtrapolationObjective, trace)
    kernel = get_retval(trace)
    noise = trace[:noise]
    train_xs, = get_args(trace)
    train_ys = trace[:ys]
    
    try
        μ, Σ = compute_predictive(kernel, noise, train_xs, train_ys, obj.extrap_xs)
        
        # neg MSE (higher = better)
        mse = mean((μ .- obj.extrap_ys).^2)
        return -mse
    catch e
        return -Inf
    end
end

"""
Task-Driven Adaptive Computation

Parameters:
- objective: The task objective to optimize (e.g., ExtrapolationObjective)
- τ: Temperature for importance distribution (lower = more focused)
- jmin: Minimum moves per particle
"""
@kwdef mutable struct TaskDrivenAC <: AllocationStrategy
    objective::TaskObjective = PosteriorObjective()
    τ::Float64 = 1.0
    jmin::Int = 1
end

"""
Main allocation function for Task-Driven AC.
Follows the adaptive computation framework, but uses task-conditioned priority instead of sensitivity.
"""
function compute_allocations(
    ac::TaskDrivenAC,
    traces::Vector,
    log_weights::Vector{Float64},
    budget::Int;
    verbose=false
)
    n = length(traces)
    n == 0 && return Int[]
    
    # Use task objective as priority
    objectives = [evaluate_objective(ac.objective, tr) for tr in traces]
    
    # allocate
    obj_min = minimum(objectives)
    obj_shift = obj_min < 0 ? -obj_min + 1e-10 : 1e-10
    priorities = [log(o + obj_shift) for o in objectives]
    importance = softmax(priorities ./ ac.τ)
    allocations = [max(round(Int, importance[i] * budget), 0) + ac.jmin for i in 1:n]
    
    return allocations
end

#==============================================================================
  SIMPLIFIED SCORE-BASED AC (previous version, kept for comparison)
==============================================================================#

@kwdef struct ScoreBasedAC <: AllocationStrategy
    τ::Float64 = 50.0
    jmin::Int = 1
end

function compute_allocations(
    ac::ScoreBasedAC,
    traces::Vector,
    log_weights::Vector{Float64},
    budget::Int;
    verbose=false
)
    n = length(traces)
    n == 0 && return Int[]
    
    scores = [get_score(trace) for trace in traces]
    importance = softmax(scores ./ ac.τ)
    allocations = [max(round(Int, importance[i] * budget), 0) + ac.jmin for i in 1:n]
    
    if verbose
        println("Scores: $(round.(scores, digits=1))")
        println("Importance (τ=$(ac.τ)): $(round.(importance, digits=3))")
        println("Allocations: $allocations (sum=$(sum(allocations)))")
    end
    
    return allocations
end


function softmax(x)
    x_shifted = x .- maximum(x)
    exp_x = exp.(x_shifted)
    return exp_x ./ sum(exp_x)
end