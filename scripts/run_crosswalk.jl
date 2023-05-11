using Turing
using POMDPs
using Distributions

include("../src/turing_models.jl")
include("../src/utils.jl")
include("../src/evaluation.jl")
include("../src/plotting.jl")
include("../environments/crosswalk.jl")

Turing.setadbackend(:zygote)

# Create POMDP, Policy, and Belief Updater
horizon = 30
v_des = 11.17
s_ego = [-15., v_des]
s_ped = [0.0, -2, 0.0, 1.6]
s0 = vcat(s_ego, s_ped)
b0 = s0
policy = IntelligentDriverModel(v_des=v_des)
up = PreviousObservationUpdater()
pomdp = CrosswalkPOMDP(dt=0.1, γ=1.0, crosswalk_width=4.0, lane_width=3.0)

# Disturbance model
σ = sqrt.(Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1]))
px = MvNormal(zeros(6), σ)

# Distance to failure
ϵ = 0.01
fail_distance = 0.5
ϕ(s) = maximum([0.0, ego_ped_distance(s) - fail_distance])

# Probabilistic model
model = pomdp_validation_model(pomdp, policy, up, px, ϕ, s0, b0, horizon, ϵ, missing)

# Sample
num_chains = 20
chain = mapreduce(c -> sample(model, NUTS(1000, 0.65, max_depth=7), 1000, discard_adapt=false), chainscat, 1:num_chains)
x_samples = Array(group(chain, :x))
failure_events = simulate_failures(pomdp, policy, up, px, ϕ, s0, b0, horizon, x_samples)

# Run PG Evaluation
pg_chain = mapreduce(c -> sample(model, PG(100), 1000), chainscat, 1:num_chains)
pg_samples = Array(group(pg_chain, :x))
pg_failures = simulate_failures(pomdp, policy, up, px, ϕ, s0, b0, horizon, pg_samples)

#Run MC Evaluation
mc_failures = montecarlo(pomdp, policy, up,  px, ϕ, b0, s0, horizon, num_chains*1000)