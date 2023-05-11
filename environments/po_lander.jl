using Random
using LinearAlgebra
using POMDPs
using POMDPTools
using Distributions
using GridInterpolations
using Flux
using BSON
using POMDPTools: ModelTools

## POMDP
struct LanderActionSpace
    min_lateral::Float64
    max_lateral::Float64
    max_thrust::Float64
    max_offset::Float64
    actions::Union{Nothing, Vector{Vector{Float64}}}
end

function LanderActionSpace(discrete::Bool, bins::Tuple{Int64, Int64, Int64})
    min_lateral = -10.0
    max_lateral = 10.0
    max_thrust = 15.0
    max_offset = 1.0
    if discrete
        lat_range = range(min_lateral, max_lateral, length=bins[1])
        lon_range = range(0.0, max_thrust, length=bins[2])
        offset_range = range(-1*max_offset, max_offset, length=bins[3])
        actions = Vector{Vector{Float64}}()
        
        # Add zero lateral thrust actions (longitudinal thrust only)
        n_lat_settings = bins[1]
        n_lon_settings = bins[2]
        n_offset_settings = bins[3]
        for i=1:n_lon_settings
            ai = Vector{Float64}([0.0, lon_range[i], 0.0])
            push!(actions, ai)
        end
        # Add non-zero lateral thrust actions
        for i = 1:n_lat_settings
            if lat_range[i] != 0.0
                for j = 1:n_lon_settings
                    for k = 1:n_offset_settings
                        aa = Vector{Float64}([lat_range[i], lon_range[j], offset_range[k]])
                        push!(actions, aa)
                    end
                end
            end
        end
        LanderActionSpace(-10.0, 10.0, 15.0, 1.0, actions)
    
    else
        return LanderActionSpace(-10.0, 10.0, 15.0, 1.0, nothing)
    end
end

struct LunarLander <: POMDP{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    discrete::Bool
    bins::Tuple{Int64, Int64, Int64} # Num action bins only used if discrete
    action_space::LanderActionSpace
    σ_s::Float64
    dt::Float64
    m::Float64 # 1000's kg
    I::Float64 # 1000's kg*m^2
    Q::Vector{Float64}
    R::Vector{Float64}
end

function LunarLander(;discrete::Bool=false, bins=(5,5,5), σ_s::Float64=0.0, dt::Float64=0.5, m::Float64=1.0, I::Float64=10.0)
    Q = [0.0, 0.0, 0.0, 0.1, 0.1, 0.01]
    R = [1.0, 0.01, 0.1]
    action_space = LanderActionSpace(discrete, bins)
    return LunarLander(discrete, bins, action_space, σ_s, dt, m, I, Q, R)
end

function Base.iterate(aspace::LanderActionSpace)
    return iterate(aspace.actions)
end

function Base.iterate(aspace::LanderActionSpace, i::Int64)
    return iterate(aspace.actions, i)
end

function Base.length(aspace::LanderActionSpace)
    return length(aspace.actions)
end

function POMDPs.actionindex(p::P, a::Vector{Float64}) where P<:POMDP
    return findfirst(x->x==a, p.action_space.actions)
end

function Base.rand(as::LanderActionSpace)
    lateral_range = as.max_lateral - as.min_lateral
    f_x = rand()*lateral_range + as.min_lateral
    f_z = rand()*as.max_thrust
    offset = (rand()-0.5)*2.0*as.max_offset
    return [f_x, f_z, offset]
end

function Base.rand(rng::AbstractRNG, as::LanderActionSpace)
    lateral_range = as.max_lateral - as.min_lateral
    f_x = rand(rng)*lateral_range + as.min_lateral
    f_z = rand(rng)*as.max_thrust
    offset = (rand()-0.5)*2.0*as.max_offset
    return [f_x, f_z, offset]
end

function update_state(m::LunarLander, s::Vector{Float64}, a::Vector{Float64})
    if s[2] < 1.0
        return s
    end
    x = s[1]
    z = s[2]
    θ = s[3]
    vx = s[4]
    vz = s[5]
    ω = s[6]

    f_lateral = a[1]
    thrust = a[2]
    δ = a[3]

    fx = cos(θ)*f_lateral - sin(θ)*thrust
    fz = cos(θ)*thrust + sin(θ)*f_lateral
    torque = -δ*f_lateral

    ax = fx/m.m
    az = fz/m.m
    ωdot = torque/m.I

    vxp = vx + ax*m.dt
    vzp = vz + (az - 9.0)*m.dt
    ωp = ω + ωdot*m.dt

    xp = x + vx*m.dt
    zp = z + vz*m.dt
    θp = θ + ω*m.dt

    sp = [xp, zp, θp, vxp, vzp, ωp]
    return sp
end

function update_state(m::LunarLander, s, a)
    x = s[1]
    z = s[2]
    θ = s[3]
    vx = s[4]
    vz = s[5]
    ω = s[6]

    f_lateral = a[1]
    thrust = a[2]
    δ = a[3]

    fx = cos(θ)*f_lateral - sin(θ)*thrust
    fz = cos(θ)*thrust + sin(θ)*f_lateral
    torque = -δ*f_lateral

    ax = fx/m.m
    az = fz/m.m
    ωdot = torque/m.I

    vxp = vx + ax*m.dt
    vzp = vz + (az - 9.0)*m.dt
    ωp = ω + ωdot*m.dt

    xp = x + vx*m.dt
    zp = z + vz*m.dt
    θp = θ + ω*m.dt

    sp = [xp, zp, θp, vxp, vzp, ωp]
    return sp
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::Vector{Float64}, mdp::LunarLander)
    v = SVector{6,Float64}(s)
    return v
end

function POMDPs.convert_s(::Type{Vector{Float64}}, v::AbstractVector{Float64}, mdp::LunarLander)
    s = Vector{Float64}(s)
end

function get_observation(s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG,
                    σz::Float64=1.0, σω::Float64=0.01, σx::Float64=0.1)
    x = [randn(rng)*σz, randn(rng)*σω, randn(rng)*σx]
    return get_observation(s, a, x)
end

function get_observation(s::Vector{Real}, a::Vector{Real}, x::Vector{Real})
    z = s[2]
    θ = s[3]
    ω = s[6]
    xdot = s[4]
    agl = z/cos(θ) + x[1]
    obsω = ω + x[2]
    obsxdot = xdot + x[3]
    o = [agl, obsω, obsxdot]
    return o
end

function get_observation(s, a, x)
    z = s[2]
    θ = s[3]
    ω = s[6]
    xdot = s[4]
    agl = z/cos(θ) + x[1]
    obsω = ω + x[2]
    obsxdot = xdot + x[3]
    o = [agl, obsω, obsxdot]
    return o
end

function get_reward(s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}; dt::Float64=0.1) 
    x = sp[1]
    z = sp[2]
    δ = abs(x)
    θ = abs(sp[3])
    vx = sp[4]
    vz = sp[5]

    if δ >= 15.0 || θ >= 0.5
        r = -1000.0
    elseif z <= 1.0
        r = -(δ + vz^2) + 100.0
    else
        r = -1.0*dt*2.0
    end
    return r
end

function POMDPs.reward(p::LunarLander, s, a, sp)
    get_reward(s, a, sp, dt=p.dt)
end

function POMDPs.transition(m::LunarLander, s, a)
    sp = update_state(m, s, a)
    return Deterministic(sp)
end

function POMDPs.gen(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, x::Vector{Float64})
    sp = update_state(m, s, a)
    o = get_observation(sp, a, x)
    r = get_reward(s, a, sp, dt=m.dt)
    return (sp=sp, o=o, r=r)
end

POMDPs.actions(p::LunarLander) = p.action_space
POMDPs.actiontype(::LunarLander) = Vector{Float64}
POMDPs.discount(::LunarLander) = 0.99

function POMDPs.initialstate_distribution(::LunarLander)
    μ = [0.0, 50.0, 0.0, 0.0, -10.0, 0.0]
    σ = [0.1, 0.1, 0.01, 0.1, 0.1, 0.01]
    σ = diagm(σ)
    return MvNormal(μ, σ)
end

function POMDPs.isterminal(::LunarLander, s::Vector{Float64})
    x = s[1]
    z = s[2]
    δ = abs(x)
    θ = abs(s[3])
    if δ >= 15.0 || θ >= 0.5 || z <= 1.0
        return true
    else
        return false
    end
end

function ModelTools.obs_weight(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}, o::Vector{Float64})
    R = [1.0, 0.01, 0.1]
    z = sp[2]
    θ = sp[3]
    ω = sp[6]
    xdot = s[4]
    agl = z/cos(θ)
    dist = MvNormal([agl, ω, xdot], R)
    return pdf(dist, o)
end

struct LanderPolicy <: Policy
    m::LunarLander
end

POMDPs.updater(p::LanderPolicy) = EKFUpdater(p.m, p.m.Q.^2, p.m.R.^2)

function POMDPs.action(p::LanderPolicy, b::MvNormal)
    s = mean(b)
    act = [-0.5*s[4] -0.5*s[5] 0.0][1,:]
    return act
end


## For EKF Belief Updater
function x2s(m::LunarLander, x::Vector{Float64})
    s = x
    return s
end

function x2s(m::LunarLander, x)
    s = x
    return s
end

function s2x(m::LunarLander, s::Vector{Float64})
    x = s
    return x
end

function s2x(m::LunarLander, s)
    x = s
    return x
end

sparsify!(x, eps) = x[abs.(x) .< eps] .= 0.0

function gen_A(m::LunarLander, s::Vector{Float64}, a::Vector{Float64})
    θ = s[3]
    f_l = a[1]
    thrust = a[2]

    A14 = m.dt
    A25 = m.dt
    A36 = m.dt
    A43 = (-sin(θ)*f_l - cos(θ)*thrust)*m.dt/m.m
    A53 = (-sin(θ)*thrust + cos(θ)*f_l)*m.dt/m.m

    A = [1.0 0.0 0.0 A14 0.0 0.0;
         0.0 1.0 0.0 0.0 A25 0.0; 
         0.0 0.0 1.0 0.0 0.0 A36;
         0.0 0.0 A43 1.0 0.0 0.0; 
         0.0 0.0 A53 0.0 1.0 0.0; 
         0.0 0.0 0.0 0.0 0.0 1.0;
         ]

    return A

end

function gen_C(m::LunarLander, s::Vector{Float64})
    z = s[2]
    θ = s[3]

    C12 = 1/(cos(θ) + eps())
    C13 = z*sin(θ)/(cos(θ)^2 + eps())
    C26 = 1.0
    C34 = 1.0

    C = [0.0 C12 C13 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 C26; 
         0.0 0.0 0.0 C34 0.0 0.0;
         ]

    return C
end

function gen_A(m::LunarLander, s, a)
    θ = s[3]
    f_l = a[1]
    thrust = a[2]

    A14 = m.dt
    A25 = m.dt
    A36 = m.dt
    A43 = (-sin(θ)*f_l - cos(θ)*thrust)*m.dt/m.m
    A53 = (-sin(θ)*thrust + cos(θ)*f_l)*m.dt/m.m

    A = [1.0 0.0 0.0 A14 0.0 0.0;
         0.0 1.0 0.0 0.0 A25 0.0; 
         0.0 0.0 1.0 0.0 0.0 A36;
         0.0 0.0 A43 1.0 0.0 0.0; 
         0.0 0.0 A53 0.0 1.0 0.0; 
         0.0 0.0 0.0 0.0 0.0 1.0;
         ]

    return A

end

function gen_C(m::LunarLander, s)
    z = s[2]
    θ = s[3]

    C12 = 1/(cos(θ) + eps())
    C13 = z*sin(θ)/(cos(θ)^2 + eps())
    C26 = 1.0
    C34 = 1.0

    C = [0.0 C12 C13 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 C26; 
         0.0 0.0 0.0 C34 0.0 0.0;
         ]

    return C
end


##EKF
struct EKFUpdater{P<:POMDP} <: Updater
    m::P
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

function EKFUpdater(m, Q::Array, R::Array)
    if ndims(Q) == 1
        Q = diagm(Q)
    end
    if ndims(R) == 1
        R = diagm(R)
    end
    EKFUpdater(m, Q, R)
end

function belief_type(::EKFUpdater)
    return MvNormal{Float64, Matrix{Float64}}
end

function POMDPs.update(up::EKFUpdater, b::B, a, o::O) where {B,O}
    μ = mean(b)
    n = length(μ)
    Σ = cov(b)
    s = x2s(up.m, μ)

    # Predict
    sp = update_state(up.m, s, a)
    z = get_observation(sp, a, zeros(3))
    
    xp = s2x(up.m, sp)

    A = gen_A(up.m, μ, a)
    C = gen_C(up.m, xp)

    Σ_hat = A*Σ*transpose(A) + up.Q

    # Update
    y = o - z

    S = C*Σ_hat*transpose(C) + up.R
    K = Σ_hat*transpose(C)/S

    μp = xp + K*y

    Σp = Matrix(Hermitian((Matrix{Float64}(I, n, n) - K*C)*Σ_hat))
    bp = MvNormal(μp, Σp)
    
    return bp
end

function Base.rand(rng, d::Deterministic)
    return d.val
end

## NN Policy
struct LanderNNPolicy{C<:Chain} <: Policy
    model::C
    xshift::Vector{Float64}
    xscale::Vector{Float64}
    yshift::Vector{Float64}
    yscale::Vector{Float64}
end

function LanderNNPolicy(network, io_transformers)
    xshift = Float64.(vec(io_transformers.μ_x))
    xscale = Float64.(vec(io_transformers.σ_x))
    yshift = Vector{Float64}([transformers.μ_y])
    yscale = Float64.(vec(transformers.σ_y))
    return LanderNNPolicy(network, xshift, xscale, yshift, yscale)
end

function POMDPs.action(policy::LanderNNPolicy, b)
    s = mean(b)
    
    # Preprocess the state
    st = Float32.((reshape(s, 6, 1) .- policy.xshift) ./ policy.xscale)

    # Call the NN model
    y = policy.model(st)

    # Postrocess action
    a = vec(Float64.(y)) .* policy.yscale .+ policy.yshift
    
    return a
end

function load_network(network_path)
    BSON.@load network_path model transformers
    return model, transformers
end

## Plotting
function plot_z_trajectory(p::LunarLander, h; fig=plot(xlabel="x", ylabel="z", title="Lander Trajectories"), label="", color=1, alpha=0.3, kwargs...)
    x = map(s->s[1], h[:s])
    z = map(s->s[2], h[:s])
    plot!(fig, x, z, label="", alpha=alpha, color=color; kwargs...)
end

function plot_trajectory(p::LunarLander, h; fig=plot(xlabel="zdot", ylabel="z", title="Lander Trajectories"), label="", color=1, alpha=0.3, kwargs...)
    zdot = map(s->s[5], h[:sp])
    z = map(s->s[2], h[:sp])
    plot!(fig, zdot, z, label="", alpha=alpha, color=color; kwargs...)
end


