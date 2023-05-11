using POMDPs
using POMDPTools
using Parameters
using Random

@with_kw mutable struct CrosswalkPOMDP <: POMDP{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    dt::Float64 = 0.1 # Simulation timestep
    γ::Float64 = 1.0 # discount
    crosswalk_width::Float64 = 1.0
    lane_width::Float64 = 3.0
end

function get_observation(s::Vector, a_ego, x::Vector)
    s_ped = s[3:end]
    return vcat(s[1:2], s_ped[1:2] .+ x[1:2], s_ped[3:4] .+ x[3:4])
end

function get_reward(s, a, sp)
    return sp[3] - s[3]
end

function ego_ped_distance(s::Vector)
    x_diff = s[1] - s[3]
    y_diff = s[4]
    d = sqrt(x_diff^2 + y_diff^2)
    return d
end

function action_ped(mdp::CrosswalkPOMDP, s, x)
    a_ped_new = x[5:6]
    return a_ped_new
end

function propagate(mdp::CrosswalkPOMDP, s::Vector{Float64})
    dt = mdp.dt
    s_ped = s[4:7]
    x_ped = s_ped[1:2]
    v_ped = s_ped[3:4]
    v_ped_new = v_ped .+ a_ped*dt
    x_ped_new = x_ped .+ v_ped*dt .+ a_ped*dt^2

    s_ego = s[1:3]
    v_ego_new = s_ego[2] + s_ego[3]*dt
    x_ego_new = s_ego[1] + s_ego[2]*dt + s_ego[3]*dt^2

    s_new = vcat(x_ego_new, v_ego_new, s_ego[3], x_ped_new, v_ped_new, a_ped)
    return s_new
end

function step(mdp::CrosswalkPOMDP, s, a_ego, a_ped)
    dt = mdp.dt
    s_ped = s[3:6]
    x_ped = s_ped[1:2]
    v_ped = s_ped[3:4]
    v_ped_new = v_ped .+ a_ped*dt
    x_ped_new = x_ped .+ v_ped*dt

    s_ego = s[1:2]
    v_ego_new = s_ego[2] + a_ego*dt
    if v_ego_new < 0
        v_ego_new = 0
        a_ego = 0
        s_ego = [s[1], 0.0]
    end
    x_ego_new = s_ego[1] + s_ego[2]*dt
    s_new = vcat(x_ego_new, v_ego_new, x_ped_new, v_ped_new)
    return s_new
end

function POMDPs.gen(p::CrosswalkPOMDP, s, a, x)
    sp = step(p, s, a, x[5:6])
    o = get_observation(sp, a, x[1:4])
    r = get_reward(s, a, sp)
    return (sp=sp, o=o, r=r)
end

@with_kw mutable struct IntelligentDriverModel <: Policy
    a::Real = 0.0 # predicted acceleration
    σ::Float64 = 0.0 # optional stdev on top of the model, set to zero or NaN for deterministic behavior
    k_spd::Float64 = 1.0 # proportional constant for speed tracking when in freeflow [s⁻¹]
    δ::Float64 = 4.0 # acceleration exponent [-]
    T::Float64  = 1.5 # desired time headway [s]
    v_des::Float64 = 29.0 # desired speed [m/s]
    s_min::Float64 = 5.0 # minimum acceptable gap [m]
    a_max::Float64 = 3.0 # maximum acceleration ability [m/s²]
    d_cmf::Float64 = 2.0 # comfortable deceleration [m/s²] (positive)
    d_max::Float64 = 9.0 # maximum deceleration [m/s²] (positive)
    lane_width::Float64 = 3.0
end

function set_desired_speed!(model::IntelligentDriverModel, v_des::Float64)
    model.v_des = v_des
    model
end

function track_longitudinal!(model::IntelligentDriverModel, v_ego, v_oth, headway)
    if headway < 0.0
        model.a = -model.d_max
    else

        Δv = v_oth - v_ego
        s_des = model.s_min + v_ego*model.T - v_ego*Δv / (2*sqrt(model.a_max*model.d_cmf))
        v_ratio = model.v_des > 0.0 ? (v_ego/model.v_des) : 1.0
        model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des/headway)^2)
    end

    model.a = clamp(model.a, -model.d_max, model.a_max)

    return model
end

function track_accel!(model::IntelligentDriverModel, v_ego)
    Δv = model.v_des - v_ego
    model.a = Δv*model.k_spd # predicted accel to match target speed
    model.a = clamp(model.a, -model.d_max, model.a_max)
    return model.a
end

reset_hidden_state!(model::IntelligentDriverModel) = model


function POMDPs.action(model::IntelligentDriverModel, o)
    x_ego  = o[1]
    vx_ego = o[2]
    x_ped  = o[3]
    vx_ped = o[5]
    
    if o[4] > -model.lane_width/2 && o[4] < model.lane_width/2 && x_ped >= x_ego
        headway = x_ped - x_ego
        track_longitudinal!(model, vx_ego, vx_ped, headway)
    else
        track_accel!(model, vx_ego)
    end

    return model.a
end

function _action(model::IntelligentDriverModel, s::Vector)
    model = track_longitudinal!(model, s_idm[1], s_idm[2], s_idm[3])
    return model.a
end

function _action(model::IntelligentDriverModel, v_ego::Float64)
    Δv = model.v_des - v_ego
    model.a = Δv*model.k_spd # predicted accel to match target speed
    model.a = clamp(model.a, -model.d_max, model.a_max)
    return model.a
end

function Base.rand(rng::AbstractRNG, model::IntelligentDriverModel)
    if isnan(model.σ) || model.σ ≤ 0.0
        return model.a
    else
        rand(rng, Normal(model.a, model.σ))
    end
end


## Plotting
function plot_ped_trajectory(p::CrosswalkPOMDP, history; fig=plot(xlabel="r_x", ylabel="r_y", title="Pedestrian Trajectories"), label="", color=1, alpha=0.3, kwargs...)
    # plot!(fig, [], color=color, label=label; kwargs...)
    x = map(s->s[3], history[:s])
    y = map(s->s[4], history[:s])
    plot!(fig, x, y, label="", alpha=alpha, color=color; kwargs...)
end

function plot_noise_trajectory(p::CrosswalkPOMDP, history; fig=plot(xlabel="x", ylabel="y", title="Observation Trajectories"), label="", color=1, alpha=0.3, kwargs...)
    # plot!(fig, [], color=color, label=label; kwargs...)
    nx = map(o->o[3], history[:o])
    ny = map(o->o[4], history[:o])
    plot!(fig, nx, ny, label="", alpha=alpha, color=color; kwargs...)
end

function plot_ego_ped_distance(p::CrosswalkPOMDP, history; fig=plot(xlabel="Timestep", ylabel="range", title="Agent Distance"), label="", color=1, alpha=0.3, kwargs...)
    r = ego_ped_distance.(history[:s])
    plot!(fig, r, label="", alpha=alpha, color=color; kwargs...)
end

rectangle(w, h, x, y) = Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function plot_ego_av_trajectory(p::CrosswalkPOMDP, history; fig=plot(xlabel="x", ylabel="y", title="Agent Distance"), label="", color=1, alpha=0.3, kwargs...)
    plot_ped_trajectory(p, history; fig=fig)
    av_x = map(s->s[1], history[:s])
    av_y = zeros(length(av_x))#map(s->s[2], history[:s])
    plot!(fig, av_x, av_y, label="", alpha=alpha, color=color; markershape=:square, kwargs...)
end