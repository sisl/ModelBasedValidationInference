using Crux
using Flux
using Distributions
using POMDPs
using Random
using POMDPTools.Policies: FunctionPolicy

include("pendulum.jl")

mdp = PendulumMDP(max_torque=2, dt=0.1)

S = state_space(mdp)
adim = 1
amin = -2*ones(Float32, adim)
amax = 2*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Build the networks
idim = S.dims[1] + adim
μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 32, tanh, init=Winit, bias=binit(32, S.dims[1])), 
                              Dense(32, 32, tanh, init=Winit, bias=binit(32, 32)), 
                              Dense(32, adim, init=Winit, bias=binit(adim, 32))))
V() = ContinuousNetwork(Chain(Dense(S.dims[1], 32, tanh, init=Winit, bias=binit(32, S.dims[1])), 
                              Dense(32, 32, init=Winit, bias=binit(32, 32)), 
                             Dense(32, 1, init=Winit, bias=binit(1, 32))))
log_std() = -0.5f0*ones(Float32, adim)

shared = (max_steps=100, N=Int(5e5), S=S)
on_policy = (ΔN=4000, 
             λ_gae=0.97, 
             a_opt=(batch_size=4000, epochs=80, optimizer=ADAM(3e-4)), 
             c_opt=(batch_size=4000, epochs=80, optimizer=ADAM(1e-3)))

𝒮_ppo = PPO(;π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), λe=0f0, shared..., on_policy...)
solve(𝒮_ppo, mdp)


𝒮_ppo.agent.π.A.μ