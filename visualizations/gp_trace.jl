include("../src/bayesianGP.jl")
using Gen
using Plots

traces = [first(generate(model, (collect(Float64, -1:0.1:1),))) for i in 1:12]
plot([visualize_gp_trace(t, -1, 1) for t in traces]...)