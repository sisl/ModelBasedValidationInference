using POMDPs
using Distributions
using LinearAlgebra


function sample_to_disturbance(sample, horizon)
    syms = sort(collect(keys(sample)))
    x = zeros(length(syms), horizon)
    for (i,sym) in enumerate(syms)
        x[i, :] = sample[sym]
    end
    return x
end

function montecarlo(p::MDP, policy, px, ϕ, s0, horizon, n)
    failure_events = Vector{FailureEvent}()
    for i=1:n
        x = rand(px, horizon)
        h = simulate(p, policy, x, s0, horizon)
        fail = ϕ.(h[:s]) .<= 0.0
        if any(fail)
            fail_idx = findfirst(fail)
            hfail = trim_history(h, fail_idx)
            xfail = x[:, 1:fail_idx]
            event = FailureEvent(hfail, xfail, sum(logpdf(px, xfail)))
            push!(failure_events, event)
        end
    end
    return failure_events
end

function montecarlo(p::POMDP, policy::Policy, up::Updater, px, ϕ, b0, s0, horizon, n)
    failure_events = Vector{FailureEvent}()
    for i=1:n
        x = rand(px, horizon)
        h = simulate(p, policy, up, x, b0, s0, horizon)
        fail = ϕ.(h[:s]) .<= 0.0
        if any(fail)
            fail_idx = findfirst(fail)
            hfail = trim_history(h, fail_idx)
            xfail = x[:, 1:fail_idx]
            event = FailureEvent(hfail, xfail, sum(logpdf(px, xfail)))
            push!(failure_events, event)
        end
    end
    return failure_events
end

function get_xt_1d(failure_events, t)
    xt =  [f.disturbance[t] for f in failure_events if length(f.disturbance) >= t]
    return reshape(xt, :, 1)
end

function get_xt(failure_events, t, dim)
    if dim ==1
        return get_xt_1d(failure_events, t)
    end
    xt =  [f.disturbance[:, t] for f in failure_events if size(f.disturbance, 2) >= t]
    return hcat(xt...)'#reshape(xt, :, dim)
end

function grid_min_dist(p, x)
    dists = [euclidean(p, xi) for xi in eachrow(x)]
    return minimum(dists, init=Inf)
end


function dispersion(failures::Vector{FailureEvent}, bounds, grid_size, horizon)
    # X grid
    grid = [-b:grid_size:b for b in bounds]

    # loop through each time step
    # loop through each X
    # calculate minimum distance from gris point to all points in time step
    dim = length(bounds)
    c = 0.0
    for t=1:horizon
        xt = get_xt(failures, t, dim)
        for p in Base.Iterators.product(grid...)
            dj = grid_min_dist(collect(p), xt)
            c += minimum([dj, grid_size])
        end
    end
    return c/(prod([length(collect(g)) for g in grid]) * grid_size * horizon)
end