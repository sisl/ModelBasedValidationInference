using Parameters
using POMDPs
using POMDPTools
using Distances
using Distributions
using DataStructures
using Interpolations


mutable struct FailureEvent
    history::NamedTuple
    disturbance::Any
    log_prob::Float64
end

histories(failure_events::Vector{FailureEvent}) = map(f->f.history, failure_events)
disturbances(failure_events::Vector{FailureEvent}) = map(f->f.disturbance, failure_events)
log_probs(failure_events::Vector{FailureEvent}) = map(f->f.log_prob, failure_events)

function POMDPs.simulate(m::POMDP, policy::Policy, up::Updater, x, b0, s0, N)
    h = (s=[], a=[], o=[], sp=[], b=[])
    i = 1
    s = s0
    b = b0
    while i <= N #!isterminal(m, s) &&
        # Advance
        a = action(policy, b)
        sp, o = @gen(:sp, :o)(m, s, a, x[:, i])
        bp = update(up, b, a, o)

        # Record
        push!(h[:s], s)
        push!(h[:sp], sp)
        push!(h[:a], a)
        push!(h[:o], o)
        push!(h[:b], b)

        s = sp
        b = bp
        i += 1 
    end

    return h
end

function POMDPs.simulate(m::MDP, policy::Policy, x, s0, N)
    h = (s=[], a=[], sp=[],)
    i = 1
    s = s0  
    while !isterminal(m, s) && i <= N
        # Advance
        a = action(policy, s)
        sp = @gen(:sp)(m, s, a, x[:, i])

        # Record
        push!(h[:s], s)
        push!(h[:sp], sp)
        push!(h[:a], a)

        s = sp
        i += 1 
    end

    return h
end

function trim_history(h, idx)
    names = keys(h)
    vals = values(h)
    new_vals = [v[1:idx] for v in vals]
    trimmed = (; zip(names, new_vals)...)
    return trimmed 
end

function simulate_failures(p::MDP, policy::Policy, px::Distribution, ϕ, s0, N, x_array; trim=true)
    failure_events = Vector{FailureEvent}()
    for x_flat in eachrow(x_array)
        x = reshape(x_flat, size(x_flat, 1) ÷ N, :)
        h = simulate(p, policy, x, s0, N)
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

function simulate_failures(p::POMDP, policy::Policy, up::Updater, px::Distribution, ϕ, s0, b0, N, x_array)
    failure_events = Vector{FailureEvent}()
    for x_flat in eachrow(x_array)
        x = reshape(x_flat, size(x_flat, 1) ÷ N, :)
        h = simulate(p, policy, up, x, b0, s0, N)
        fail = ϕ.(h[:sp]) .<= 0.0
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

function trajectory_pairwise_distance(trajectory_list)
    completed_pairs = DefaultDict{Int, Vector{Int}}([])
    distances = Vector{Float64}()
    for i in eachindex(trajectory_list)
        for j in eachindex(trajectory_list)
            if i != j && !(j in completed_pairs[i]) && !(i in completed_pairs[j])
                dist = trajectory_distance(trajectory_list[i], trajectory_list[j])
                push!(distances, dist)
                push!(completed_pairs[i], j)
            end
        end
    end
    return distances
end

function trajectory_distance(t1, t2)
    n1, n2 = normalize_trajectories(t1, t2)
    d = mean([euclidean(s[1], s[2]) for s in zip(n1, n2)])
    return d
end

function normalize_trajectories(t1, t2)
    # find shortest trajectory
    # interpolate each dimesnion
    t1short = true
    if length(t1) == length(t2)
        return t1, t2
    elseif length(t1) < length(t2)
        tshort = t1
        tlong = t2
    else
        tshort = t2
        tlong = t1
        t1short = false
    end

    s_matrix = hcat(tshort...)
    D = size(s_matrix, 1)
    snew = zeros(D, length(tlong))
    for i=1:D
        time = 1:1:size(s_matrix, 2)
        interp_linear = linear_interpolation(time, s_matrix[i, :])
        interp_time = range(start=1, stop=length(tshort), length=length(tlong))
        snew[i, :] = interp_linear.(interp_time)
    end
    return slicematrix(snew), tlong
end

function slicematrix(A::AbstractMatrix)
    return [A[:, i] for i in 1:size(A,2)]
end