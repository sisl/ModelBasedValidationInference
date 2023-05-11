using Turing
using POMDPs
using POMDPTools
using Flux
using Zygote
using MCMCChains
using Optim
using BSON: @load

include("../src/turing_models.jl")
include("../src/utils.jl")
include("../src/evaluation.jl")
include("../src/plotting.jl")
include("../environments/po_lander.jl")

Turing.setadbackend(:zygote)

# Create POMDP
horizon = 15
pomdp = LunarLander(dt=0.5)

# Create EKF
up = EKFUpdater(pomdp, pomdp.Q.^2, pomdp.R.^2)

# Load NN Policy
network_path = joinpath(@__DIR__, "../environments/po_lander_policy.bson")
network, transformers = load_network(network_path)
policy = LanderNNPolicy(network, transformers)

# Disturbance model
# x is 3xn
# [σz, σω, σxdot]
σ = Vector{Float64}([1.0, 0.02, 0.1])
px = MvNormal(zeros(3), σ)

# Distance function
ϕ(s) = (clamp(s[2]-1.0, 0.0, Inf) +  clamp(s[5] + 6.0, 0.0, Inf))

# Probabilistic model
s0 = Vector{Float64}([0.0, 50.0, 0.0, 0.0, -10.0, 0.0])
b0 = MvNormal(s0, [1.0, 1.0, 0.1, 0.1, 0.1, 0.01])
ϵ = 0.2

model = pomdp_validation(pomdp, policy, up, px, ϕ, s0, b0, horizon, ϵ, missing)

num_chains = 30
map_estimates = [optimize(model, MAP(), ParticleSwarm()) for i=1:num_chains]
init_params = map(m->reshape(m.values.array, 3, :), map_estimates)
chain = mapreduce(c->sample(model, NUTS(1000, 0.6, max_depth=7), 1000, discard_adapt=false, init_params=init_params[c]), chainscat, 1:num_chains)

x_samples = Array(group(chain, :x))
failure_events = simulate_failures(pomdp, policy, up, px, ϕ, s0, b0, horizon, x_samples)

# Run PG Evaluation
pg_chain = mapreduce(c -> sample(model, PG(1000), 1000), chainscat, 1:num_chains)
pg_samples = Array(group(pg_chain, :x))
pg_failures = simulate_failures(pomdp, policy, up, px, ϕ, s0, b0, horizon, pg_samples)

#Run MC Evaluation
mc_failures = montecarlo(pomdp, policy, up,  px, ϕ, b0, s0, horizon, num_chains*1000)