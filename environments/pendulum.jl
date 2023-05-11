using Random
using Parameters
using POMDPs
using Plots
using POMDPTools.POMDPDistributions: ImplicitDistribution, Deterministic

@with_kw mutable struct PendulumMDP <: MDP{Tuple{Float64,Float64},Float64}
    g::Float64 = 9.81
    m::Float64 = 1.
    l::Float64 = 1.
    dt::Float64 = 0.1
    discount::Float64 = 0.99
    cost::Float64 = -100.
    max_torque::Float64 = 2.
    max_tsteps::Float64 = 100
end

initialstate(ip::PendulumMDP) = ImplicitDistribution(rng -> ((rand(rng)-0.5)*1.2, (rand(rng)-0.5)*0.2, ))

function reward(ip::PendulumMDP,
              s::Tuple{Real,Real},
              a::Real,
              sp::Tuple{Real,Real})

    if isterminal(ip, sp)
        costs = 101
    else
        costs = angle_normalize(s[1])^2 + 0.1f0 * s[2]^2 + 1f0 * a^2
    end
    return 1.0 - costs

end

angle_normalize(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π
discount(ip::PendulumMDP) = ip.discount
isterminal(::PendulumMDP, s::Tuple{Real,Real}) = abs(s[1]) > pi/2.

function euler(p::PendulumMDP, s::Tuple{Real, Real}, a::Real)
    g = p.g
    m = p.m
    l = p.l
    dt = p.dt

    a = clamp(a, -mdp.max_torque, mdp.max_torque)
    new_θdot = s[2] + (3*g/(2*l)*sin(s[1]) + 3.0/(m*l^2)*a)*dt
    new_θ = s[1] + new_θdot*dt

    return (new_θ, new_θdot)::Tuple{Real, Real}
end

function transition(p::PendulumMDP, s::Tuple{Real, Real}, a::Real)
    return Deterministic(euler(p, s, a))
end

function POMDPs.gen(mdp::PendulumMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; )
    t = transition(mdp, s, a)
    sp=rand(t)
    r = reward(mdp, s, a, sp)
    return (sp=sp, r=Float32(r))
end

function transition(mdp::PendulumMDP, s, a, x)
    sp = euler(mdp, s, a+x[1])
    return sp
end

# Adversarial gen
function POMDPs.gen(mdp::PendulumMDP, s, a, x)
    sp = euler(mdp, s, a+x[1])
    r = reward(mdp, s, a, sp)
    return (sp=sp, r=r)
end

function convert_s(::Type{A}, s::Tuple{Float64,Float64}, ip::PendulumMDP) where A<:AbstractArray
    v = copyto!(Array{Float64}(undef, 2), s)
    return v
end

function convert_s(::Type{Tuple{Float64,Float64}}, s::A, ip::PendulumMDP) where A<:AbstractArray
    return (s[1], s[2])
end

## Plotting
function plot_trajectory(p::PendulumMDP, history; fig=plot(xlabel="Timestep", ylabel="θ", title="Pendulum Trajectories"), label="", color=1, alpha=0.3, kwargs...)
    plot!(fig, map(s->s[1], history[:s]), label="", alpha=alpha, color=color; kwargs...)
end