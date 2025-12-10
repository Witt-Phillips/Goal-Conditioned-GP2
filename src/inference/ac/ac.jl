using Gen: logsumexp

abstract type AllocationStrategy end

# Default implementations for strategies that don't track state
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
  
  Following the AC paper's framework:
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
  EXTRAPOLATION OBJECTIVES
  
  These objectives focus on prediction quality *beyond* the training range,
  which is often the key task for GP kernel discovery.
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
        # Get predictive mean at extrapolation points
        μ, Σ = compute_predictive(kernel, noise, train_xs, train_ys, obj.extrap_xs)
        
        # Negative MSE (higher = better)
        mse = mean((μ .- obj.extrap_ys).^2)
        return -mse
    catch e
        return -Inf
    end
end

"""
Extrapolation uncertainty objective (no ground truth needed).

Rewards kernels that make confident (low-variance) predictions
beyond the training range. The intuition: kernels that "understand"
the underlying structure should extrapolate confidently.

Parameters:
- extrap_xs: X values beyond training range (auto-generated if not provided)
- n_extrap_points: Number of extrapolation points to generate
- extrap_range: How far beyond training data to extrapolate (fraction of data range)

Note: This objective rewards LOW uncertainty, which may not always be desirable.
For some tasks, you might want high uncertainty on extrapolation (epistemic humility).
"""
@kwdef struct ExtrapolationUncertaintyObjective <: TaskObjective
    extrap_xs::Union{Vector{Float64}, Nothing} = nothing
    n_extrap_points::Int = 10
    extrap_range::Float64 = 0.5  # extrapolate 50% beyond data range
end

function evaluate_objective(obj::ExtrapolationUncertaintyObjective, trace)
    kernel = get_retval(trace)
    noise = trace[:noise]
    train_xs, = get_args(trace)
    train_ys = trace[:ys]
    
    # Generate extrapolation points if not provided
    extrap_xs = if obj.extrap_xs !== nothing
        obj.extrap_xs
    else
        x_min, x_max = extrema(train_xs)
        x_range = x_max - x_min
        extrap_dist = x_range * obj.extrap_range
        
        # Extrapolate on both sides
        left_xs = collect(range(x_min - extrap_dist, x_min - 0.01, length=obj.n_extrap_points ÷ 2))
        right_xs = collect(range(x_max + 0.01, x_max + extrap_dist, length=obj.n_extrap_points ÷ 2))
        vcat(left_xs, right_xs)
    end
    
    try
        # Get predictive distribution at extrapolation points
        μ, Σ = compute_predictive(kernel, noise, train_xs, train_ys, extrap_xs)
        
        # Average predictive variance (diagonal of covariance matrix)
        avg_variance = mean(diag(Σ))
        
        # Return negative variance (higher = better = lower uncertainty)
        return -avg_variance
    catch e
        return -Inf
    end
end

"""
Extrapolation sensitivity objective.

Measures how much predictions change at extrapolation points
when the kernel structure changes. High sensitivity = the kernel
structure really matters for extrapolation.

This is useful for identifying which particles have "fragile" 
extrapolation behavior that might flip with small MCMC changes.
"""
@kwdef struct ExtrapolationSensitivityObjective <: TaskObjective
    extrap_xs::Union{Vector{Float64}, Nothing} = nothing
    n_extrap_points::Int = 10
    extrap_range::Float64 = 0.5
    # Store previous predictions for sensitivity calculation
    prev_predictions::Dict{UInt64, Vector{Float64}} = Dict{UInt64, Vector{Float64}}()
end

function evaluate_objective(obj::ExtrapolationSensitivityObjective, trace)
    kernel = get_retval(trace)
    noise = trace[:noise]
    train_xs, = get_args(trace)
    train_ys = trace[:ys]
    
    # Generate extrapolation points if not provided
    extrap_xs = if obj.extrap_xs !== nothing
        obj.extrap_xs
    else
        x_min, x_max = extrema(train_xs)
        x_range = x_max - x_min
        extrap_dist = x_range * obj.extrap_range
        right_xs = collect(range(x_max + 0.01, x_max + extrap_dist, length=obj.n_extrap_points))
        right_xs
    end
    
    try
        μ, _ = compute_predictive(kernel, noise, train_xs, train_ys, extrap_xs)
        
        # Use trace id to track predictions
        trace_id = objectid(trace)
        
        if haskey(obj.prev_predictions, trace_id)
            prev_μ = obj.prev_predictions[trace_id]
            # Sensitivity = how much did predictions change?
            sensitivity = mean(abs.(μ .- prev_μ))
            obj.prev_predictions[trace_id] = μ
            return sensitivity
        else
            obj.prev_predictions[trace_id] = μ
            return 0.0  # No previous prediction to compare
        end
    catch e
        return 0.0
    end
end

# Helper for diagonal extraction
diag(M::Matrix) = [M[i,i] for i in 1:size(M,1)]


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
    # Task objective
    objective::TaskObjective = PosteriorObjective()
    
    # Temperature for importance distribution
    τ::Float64 = 1.0
    
    # Minimum moves per particle
    jmin::Int = 1
    
    # Number of probe moves to estimate sensitivity
    n_probe_moves::Int = 3
    
    # Decision relevance mode
    # :competitive - 4w(1-w), peaks at w=0.5 (like MOT)
    # :weight - w, favor high-weight particles
    # :exploration - 1-w, favor low-weight particles  
    # :uniform - ignore weights
    δπ_mode::Symbol = :competitive
    
    # Arousal mode
    # :fixed - use budget as-is
    # :adaptive - scale budget based on aggregate sensitivity
    arousal_mode::Symbol = :fixed
    x0::Float64 = 5.0   # arousal intercept
    m::Float64 = 1.0    # arousal slope
    α::Int = 200        # max arousal cap
    
    # State for tracking (used for acceptance-based sensitivity)
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
    
    # Do probe moves and measure objective change
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
    
    # Sensitivity = average absolute change in objective
    # High sensitivity = MCMC moves significantly affect the task
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
        else  # :uniform
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
    
    # Normalize weights
    log_Z = logsumexp(log_weights)
    weights = exp.(log_weights .- log_Z)
    
    # --- Compute δˢ (Sensitivity) ---
    # Option 1: From acceptance history (if we've been tracking)
    # Option 2: From score variance across particles (proxy for "unsettledness")
    
    if !isempty(ac.move_counts) && sum(ac.move_counts) > 0
        # Use acceptance rates as sensitivity proxy
        # High acceptance = still exploring = needs more moves
        δS = zeros(n)
        for i in 1:n
            if i <= length(ac.move_counts) && ac.move_counts[i] > 0
                δS[i] = ac.acceptance_counts[i] / ac.move_counts[i]
            else
                δS[i] = 0.5  # default
            end
        end
    else
        # Use objective-based sensitivity
        # Particles with scores far from the mean are "interesting"
        objectives = [evaluate_objective(ac.objective, tr) for tr in traces]
        obj_mean = mean(objectives)
        obj_std = std(objectives)
        if obj_std > 0
            # Normalized distance from mean (both high and low are interesting)
            δS = [abs(o - obj_mean) / obj_std for o in objectives]
        else
            δS = ones(n)
        end
    end
    
    # --- Compute δᵖ (Decision Relevance) ---
    δπ = compute_decision_relevance(ac, weights)
    
    # --- Compute Task Relevance Δ ---
    # Combine sensitivity and decision relevance
    # Both are important: we want particles that are both
    # (a) sensitive to MCMC moves and (b) decision-relevant
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
