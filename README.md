# Goal-Conditioned GP: Automatic Gaussian Process Kernel Discovery

A Julia framework for **automatic Gaussian Process kernel discovery** using probabilistic programming. This project combines compositional kernel search with adaptive computation strategies, built on the [Gen](https://www.gen.dev/) probabilistic programming language.

## Overview

Gaussian Processes (GPs) are powerful non-parametric models, but their performance critically depends on choosing the right covariance kernel. This project automates kernel discovery by:

1. **Composing kernels** from a grammar of primitive kernels (Constant, Linear, Squared Exponential, Periodic) and composition operators (Plus, Times)
2. **Searching the space** of possible kernel structures using MCMC and particle filter methods
3. **Allocating compute intelligently** using task-driven adaptive computation

## Project Structure

```
Goal-Conditioned-GP2/
├── src/
│   ├── kernel.jl          # Kernel DSL: primitives + composition
│   ├── kernel_prior.jl    # Prior over kernel structures
│   ├── covariance.jl      # Covariance matrix computation & prediction
│   ├── model.jl           # Full GP generative model
│   ├── data_gen.jl        # Synthetic data generation from function trees
│   └── inference/
│       ├── involutive_mcmc.jl  # MCMC moves for kernel structure
│       ├── particle_filter.jl  # SMC with MCMC rejuvenation
│       └── ac/
│           └── ac.jl           # Adaptive computation strategies
├── visualizations/
│   ├── viz_funcs.jl       # MCMC visualization utilities
│   └── gp_trace.jl        # GP trace visualization
├── demos/                 # Jupyter notebooks demonstrating the system
│   ├── classic_autogp.ipynb     # Basic MCMC kernel discovery
│   ├── particle_filter.ipynb    # Particle filter inference
│   ├── ac.ipynb                 # Adaptive computation explained
│   ├── airline.ipynb            # Real data: airline passengers
│   ├── airline_pf.ipynb         # Airline data with particle filter
│   └── data_gen.ipynb           # Synthetic data generation
├── data/
│   └── tsdl.161.csv       # Airline passenger dataset (1949-1960)
└── Project.toml           # Julia dependencies
```

## Core Components

### 1. Kernel DSL (`src/kernel.jl`)

Defines a tree-structured domain-specific language for composing covariance kernels:

**Primitive Kernels:**
- `Constant(c)` — constant covariance
- `Linear(p)` — linear kernel with offset `p`
- `SquaredExponential(ℓ)` — RBF/SE kernel with length scale `ℓ`
- `Periodic(s, p)` — periodic kernel with scale `s` and period `p`

**Composition Operators:**
- `Plus(left, right)` — additive combination: `k₁(x,x') + k₂(x,x')`
- `Times(left, right)` — multiplicative combination: `k₁(x,x') × k₂(x,x')`

This allows expressing complex kernels like "linear trend + periodic seasonality":
```julia
Times(Linear(0.5), Plus(Constant(1.0), Periodic(0.3, 0.1)))
```

### 2. Kernel Prior (`src/kernel_prior.jl`)

A probabilistic generative model over kernel structures:

```julia
@gen function covariance_prior()
    kernel_type ~ choose_kernel_type()  # {Constant, Linear, SE, Periodic, Plus, Times}
    
    if kernel_type ∈ [Plus, Times]
        # Recursively generate subtrees
        return kernel_type({:left} ~ covariance_prior(), {:right} ~ covariance_prior())
    else
        # Generate kernel parameters
        return kernel_type({:param} ~ uniform(0, 1))
    end
end
```

### 3. GP Model (`src/model.jl`)

The full generative model combines kernel structure with observations:

```julia
@gen function model(xs::Vector{Float64})
    covariance_fn = {:tree} ~ covariance_prior()    # Sample kernel structure
    noise ~ gamma_bounded_below(1, 1, 0.01)         # Sample noise level
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    ys ~ mvnormal(zeros(length(xs)), cov_matrix)    # Sample observations
    return covariance_fn
end
```

### 4. Covariance Utilities (`src/covariance.jl`)

Functions for computing covariance matrices and GP predictions:

- `compute_cov_matrix(kernel, noise, xs)` — build covariance matrix
- `compute_predictive(kernel, noise, xs, ys, new_xs)` — conditional mean/covariance
- `predict_ys(kernel, noise, xs, ys, new_xs)` — sample from predictive distribution

### 5. Inference Methods

#### MCMC with Involutive Moves (`src/inference/involutive_mcmc.jl`)

Uses subtree regeneration for structure search:

1. Randomly walk down the kernel tree to select a subtree
2. Propose replacing it with a freshly sampled subtree
3. Accept/reject via Metropolis-Hastings

```julia
# Random path selection
@gen function random_node_path(n::Kernel)
    if {:stop} ~ bernoulli(is_primitive ? 1.0 : 0.5)
        return :tree
    else
        direction = {:left} ~ bernoulli(0.5) ? :left : :right
        rest_of_path ~ random_node_path(next_node)
        return :tree => direction => rest_of_path
    end
end
```

#### Particle Filter (`src/inference/particle_filter.jl`)

Sequential Monte Carlo with MCMC rejuvenation:

```julia
function gp_particle_filter(xs, ys, num_particles, num_rounds, num_mcmc_moves, num_samples;
                           allocation=UniformAllocation(), budget=100)
    state = initialize_particle_filter(model, (xs,), obs, num_particles)
    
    for round in 1:num_rounds
        allocations = compute_allocations(allocation, state.traces, state.log_weights, budget)
        maybe_resample!(state)
        
        # Rejuvenate each particle
        for i in 1:num_particles
            for _ in 1:allocations[i]
                state.traces[i], = mh(state.traces[i], subtree_proposal, subtree_involution)
                state.traces[i], = mh(state.traces[i], select(:noise))
            end
        end
    end
    
    return sample_unweighted_traces(state, num_samples)
end
```

#### Adaptive Computation (`src/inference/ac/ac.jl`)

Intelligent compute allocation based on task relevance. Implements the framework from adaptive computation research:

**Key Concepts:**

| Symbol | Name | Description |
|--------|------|-------------|
| δˢ | Sensitivity | How much do MCMC moves change the task objective? |
| δᵖ | Decision Relevance | How important is this particle for the final decision? |
| Δ | Task Relevance | Combined score: `log(δˢ) + log(δᵖ)` |
| Arousal | Total Budget | Overall compute to spend (can be adaptive) |
| Importance | Distribution | How to distribute compute across particles |

**Allocation Strategies:**

```julia
# Uniform: equal allocation to all particles
UniformAllocation()

# Score-based: allocate based on trace log-probability
ScoreBasedAC(τ=50.0, jmin=1)

# Task-driven: allocate based on task objectives
TaskDrivenAC(
    objective = ExtrapolationObjective(extrap_xs, extrap_ys),
    τ = 1.0,           # Temperature for importance distribution
    jmin = 1,          # Minimum moves per particle
    δπ_mode = :competitive,  # Decision relevance formula
    arousal_mode = :fixed    # or :adaptive
)
```

**Decision Relevance Modes:**
- `:competitive` — `4w(1-w)`: peaks at w=0.5 (competitive hypotheses)
- `:weight` — `w`: favor high-weight particles
- `:exploration` — `1-w`: favor low-weight particles
- `:uniform` — ignore weights

**Task Objectives:**
- `PosteriorObjective()` — use log-posterior as objective
- `PredictionObjective(held_out_xs, held_out_ys)` — predictive accuracy
- `ExtrapolationObjective(extrap_xs, extrap_ys)` — extrapolation MSE
- `ExtrapolationUncertaintyObjective()` — reward confident extrapolations

### 6. Data Generation (`src/data_gen.jl`)

Generates synthetic data from function trees for testing:

```julia
# Function tree DSL (similar to kernel DSL)
Lin(slope, intercept)  # Linear: slope*x + intercept
Sin(amplitude, period) # Sine wave
Add(left, right)       # Addition
Mul(left, right)       # Multiplication

# Sample and evaluate
fn = function_prior()    # Sample random function tree
xs, ys, fn, (xmin, xmax) = get_data(fn, n_points)  # Generate data
```

## Usage Examples

### Basic MCMC Inference

```julia
include("src/inference/involutive_mcmc.jl")

# Initialize from data
trace = initialize_trace(xs, ys)

# Run MCMC
for iter in 1:1000
    trace, = mh(trace, regen_random_subtree_randomness, (), subtree_involution)
    trace, = mh(trace, select(:noise))
end

# Get discovered kernel
kernel = get_retval(trace)
println("Discovered kernel: $kernel")
```

### Particle Filter with Adaptive Computation

```julia
include("src/inference/particle_filter.jl")

# Task-driven allocation focusing on extrapolation
ac = TaskDrivenAC(
    objective = ExtrapolationObjective(extrap_xs, extrap_ys),
    τ = 1.0,
    δπ_mode = :competitive
)

traces = gp_particle_filter(
    xs, ys,
    10,              # num_particles
    5,               # num_rounds
    10,              # num_mcmc_moves (deprecated, use budget)
    5,               # num_samples to return
    ac,              # allocation strategy
    50;              # total MCMC budget per round
    verbose=true
)

# Examine results
for tr in traces
    println("Kernel: $(get_retval(tr)), Score: $(get_score(tr))")
end
```

### Real Data: Airline Passengers

```julia
import CSV
using DataFrames

# Load and normalize
df = DataFrame(CSV.File("data/tsdl.161.csv"))
xs = normalize_to_unit(df.date)
ys = center_and_scale(df.passengers)

# Run inference
traces = gp_particle_filter(xs, ys, 10, 10, 10, 5)

# The airline data exhibits: linear trend + yearly seasonality
# Expected kernel: something like Times(Linear, Periodic) or Plus(Linear, Periodic)
```

## Dependencies

Specified in `Project.toml`:

```toml
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Gen = "ea4f424c-a589-11e8-07c0-fd5c91b9da4a"
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Goal-Conditioned-GP2

# Start Julia and activate the project
julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()
```

## Demo Notebooks

| Notebook | Description |
|----------|-------------|
| `demos/classic_autogp.ipynb` | Basic MCMC kernel discovery with visualization |
| `demos/particle_filter.ipynb` | Particle filter inference on synthetic data |
| `demos/ac.ipynb` | Interactive explanation of adaptive computation |
| `demos/airline.ipynb` | MCMC on real airline passenger data |
| `demos/airline_pf.ipynb` | Particle filter on airline data |
| `demos/data_gen.ipynb` | Synthetic data generation examples |
| `demos/particle_filter_animate.ipynb` | Animated inference visualization |

## Key Ideas

### Why Compositional Kernels?

Instead of choosing from a fixed set of kernels, compositional search:
- **Discovers structure** in data (trends, periodicity, interactions)
- **Provides interpretability** — the kernel tree explains what patterns were found
- **Handles complex data** — arbitrary combinations of simple patterns

### Why Adaptive Computation?

Not all particles need equal compute:
- **Converged particles** (low acceptance rate) need fewer MCMC moves
- **Competitive particles** (middling weights) are decision-relevant
- **High-sensitivity particles** (where moves change predictions) benefit from more exploration

The AC framework allocates compute where it matters most for the inference task.

## References

This project draws on ideas from:

- **AutoGP.jl** — Compositional kernel search for Gaussian Processes
- **Gen.jl** — Probabilistic programming with programmable inference
- **Adaptive Computation** — Task-driven resource allocation for inference

AI tools like Cursor were used to generate code.

## License

See LICENSE file for details.
