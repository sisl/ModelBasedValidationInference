using Turing
using POMDPs
using POMDPTools
using Flux
using Crux
using MCMCChains
using BSON: @load

include("../src/turing_models.jl")
include("../src/utils.jl")
include("../src/evaluation.jl")
include("../src/plotting.jl")
include("../environments/pendulum.jl")

Turing.setadbackend(:zygote)

# Create Pendulum MDP
horizon = 50
s0 = Tuple{Float64, Float64}((0.0, 0.0))
mdp = PendulumMDP(max_torque=2, dt=0.1)

# Load policy
@load joinpath(@__DIR__,"../environments/pendulum_ppo_model.bson") model
model |> gpu
policy = FunctionPolicy(s->model(collect(s))[1])

#Disturbance Model 
σ = sqrt(0.1)
px = MvNormal(1, σ)

# Distance to failure function
ϵ = 0.05
fail_threshold = 0.5
ϕ(s) = 1.0*(maximum([0.0, fail_threshold - abs(s[1])])) 

# Probabilistic model
pendulum_model = mdp_validation_model(mdp, policy, px, ϕ, s0, horizon, ϵ, missing)

# Sample
num_chains = 10
init_params = [rand(Distributions.Uniform(-2, 2), horizon) for i=1:num_chains]
chain = mapreduce(c -> sample(pendulum_model, Turing.NUTS(1000, 0.65, max_depth=6), 1000, discard_adapt=false, init_params=init_params[c]), chainscat, 1:num_chains)
x_samples = Array(group(chain, :x))
failure_events = simulate_failures(mdp, policy, px, ϕ, s0, horizon, x_samples)

plt = plot_failures(mdp, Vector{FailureEvent}(failure_events), plot_trajectory)

# Run PG Evaluation
pg_chain = mapreduce(c -> sample(pendulum_model, PG(100), 1000, init_params=init_params[c]), chainscat, 1:num_chains)
pg_samples = Array(group(pg_chain, :x))
pg_failures = simulate_failures(mdp, policy, px, ϕ, s0, horizon, pg_samples)


# Run MC Evaluation
mc_failures = montecarlo(mdp, policy,  px, ϕ, s0, horizon, num_chains*1000)