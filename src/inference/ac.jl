#= Adapted from the AC codebase: https://github.com/CNCLgithub/AdaptiveComputation =#

using Gen: logsumexp

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
  
  Following AC framework:
  - δˢ (Sensitivity): How much do MCMC moves change the task objective?
  - δᵖ (Decision Relevance): How much does this particle matter for decisions?
  - Δ (Task Relevance): Combined score determining allocation
  - Arousal: Total compute budget (can be adaptive)
  - Importance: Distribution of compute across particles
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
This is a fallback when no task-specific objective is defined.
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
Use this when you have held-out extrapolation data.

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

Allocates compute based on:
- δˢ: Sensitivity - how much do MCMC moves change the objective?
- δᵖ: Decision relevance - how competitive is this particle?

Parameters:
- objective: The task objective to optimize
- τ: Temperature for importance distribution
- jmin: Minimum moves per particle
- n_probe_moves: Number of trial moves to estimate sensitivity
- δπ_mode: How to compute decision relevance (:competitive, :weight, :uniform)
- arousal_mode: How to set total budget (:fixed, :adaptive)
- x0, m, α: Arousal parameters (for adaptive mode)
"""
@kwdef mutable struct TaskDrivenAC <: AllocationStrategy
    objective::TaskObjective = PosteriorObjective()
    τ::Float64 = 1.0
    jmin::Int = 1
    n_probe_moves::Int = 3
    δπ_mode::Symbol = :competitive
    arousal_mode::Symbol = :fixed
    x0::Float64 = 5.0   # arousal intercept
    m::Float64 = 1.0    # arousal slope
    α::Int = 200        # max arousal cap
    acceptance_counts::Vector{Int} = Int[]
    move_counts::Vector{Int} = Int[]
end

function reset_round!(ac::TaskDrivenAC, n::Int)
    ac.acceptance_counts = zeros(Int, n)
    ac.move_counts = zeros(Int, n)
end

function record_move!(ac::TaskDrivenAC, i::Int, accepted::Bool)
    if i <= length(ac.move_counts)
        ac.move_counts[i] += 1
        if accepted
            ac.acceptance_counts[i] += 1
        end
    end
end

"""
Estimate sensitivity δˢ for a particle by doing probe MCMC moves
and measuring how much the objective changes.
"""
function estimate_sensitivity(
    ac::TaskDrivenAC,
    trace,
    mcmc_kernel::Function,
    mcmc_args::Tuple
)
    P_before = evaluate_objective(ac.objective, trace)
    
    total_change = 0.0
    current_trace = trace
    n_accepted = 0
    
    for _ in 1:ac.n_probe_moves
        new_trace, accepted = mcmc_kernel(current_trace, mcmc_args...)
        if accepted
            n_accepted += 1
            P_after = evaluate_objective(ac.objective, new_trace)
            total_change += abs(P_after - P_before)
            current_trace = new_trace
            P_before = P_after
        end
    end
    
    # sensitivity = average absolute change in objective
    δS = ac.n_probe_moves > 0 ? total_change / ac.n_probe_moves : 0.0
    
    return δS, current_trace, n_accepted
end

"""
Compute decision relevance δᵖ based on particle weights.
"""
function compute_decision_relevance(ac::TaskDrivenAC, weights::Vector{Float64})
    n = length(weights)
    δπ = zeros(n)
    
    for i in 1:n
        w = weights[i]
        δπ[i] = if ac.δπ_mode == :competitive
            # Peaks at w=0.5 - particles near decision boundary
            4 * w * (1 - w)
        elseif ac.δπ_mode == :weight
            # Favor high-weight particles
            w
        elseif ac.δπ_mode == :exploration
            # Favor low-weight particles (exploration)
            1 - w
        else  # uniform
            1.0 / n
        end
    end
    
    return δπ
end

"""
Main allocation function for Task-Driven AC.

This version uses a simpler approach that doesn't require MCMC kernel access:
- δˢ is approximated from acceptance history (if available) or score variance
- δᵖ comes from weights
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
    
    log_Z = logsumexp(log_weights)
    weights = exp.(log_weights .- log_Z)
    
    # --- Compute δˢ (Sensitivity) ---
    # Use acceptance rates as sensitivity proxy
    #     # High acceptance = still exploring = needs more moves
    #     δS = zeros(n)
    #     for i in 1:n
    #         if i <= length(ac.move_counts) && ac.move_counts[i] > 0
    #             δS[i] = ac.acceptance_counts[i] / ac.move_counts[i]
    #         else
    #             δS[i] = 0.5  # default
    #         end
    #     end
    # else

    # NEW: task-conditioned sensitivity (previously used acceptance rates)
    objectives = [evaluate_objective(ac.objective, tr) for tr in traces]
    obj_min = minimum(objectives)
    obj_max = maximum(objectives)
    obj_range = obj_max - obj_min
    if obj_range > 0
        # better particles get more sensitivity
        δS = [(o - obj_min) / obj_range for o in objectives]
    else
        δS = ones(n)
    end
    
    
    # --- Compute δᵖ (Decision Relevance) ---
    δπ = compute_decision_relevance(ac, weights)
    
    # --- Compute Task Relevance Δ ---
    # Combine sensitivity and decision relevance
    Δ = [log(δS[i] + 1e-10) + log(δπ[i] + 1e-10) for i in 1:n]
    
    # --- Compute Arousal (total budget) ---
    if ac.arousal_mode == :adaptive
        # Total relevance determines how much compute we need
        total_relevance = logsumexp(Δ)
        arousal = clamp(round(Int, ac.m * (ac.x0 + total_relevance)), 0, ac.α)
    else
        arousal = budget
    end
    
    # --- Compute Importance (distribution across particles) ---
    importance = softmax(Δ ./ ac.τ)
    
    # --- Allocate ---
    allocations = [max(round(Int, importance[i] * arousal), 0) + ac.jmin for i in 1:n]
    
    if verbose
        println("=== Task-Driven AC ===")
        println("Weights: $(round.(weights, digits=3))")
        println("δˢ (sensitivity): $(round.(δS, digits=3))")
        println("δᵖ (decision relevance): $(round.(δπ, digits=3))")
        println("Δ (task relevance): $(round.(Δ, digits=2))")
        println("Importance: $(round.(importance, digits=3))")
        println("Arousal: $arousal, Allocations: $allocations (sum=$(sum(allocations)))")
    end
    
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

#==============================================================================
  UTILITIES
==============================================================================#

function softmax(x)
    x_shifted = x .- maximum(x)
    exp_x = exp.(x_shifted)
    return exp_x ./ sum(exp_x)
end

# Simple mean/std for when Statistics isn't loaded
mean(x) = sum(x) / length(x)
function std(x)
    m = mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

# Helper for diagonal extraction
diag(M::Matrix) = [M[i,i] for i in 1:size(M,1)]