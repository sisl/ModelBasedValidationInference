using POMDPs
using Turing
using Zygote

@model function mdp_validation_model(p::MDP, policy::Policy, px::Distribution, ϕ, s0, horizon, ϵ, x=missing)
    x ~ filldist(px, horizon)
    s = s0
    τ_buf = Zygote.Buffer(zeros(horizon))
    for t = 1:horizon
        τ_buf[t] = ϕ(s)
        a = action(policy, s)
        sp = @gen(:sp)(p, s, a, x[:, t])
        s = sp
    end
    τ = copy(τ_buf)
    d = minimum(τ)
    Turing.@addlogprob! logpdf(Normal(0.0, ϵ), d)
end

@model function pomdp_validation_model(p::POMDP, policy::Policy, up::Updater, px::Distribution, ϕ, s0, b0, horizon, ϵ, x=missing)
    x ~ filldist(px, horizon)
    s = s0
    b = b0
    τ_buf = Zygote.Buffer(zeros(horizon))
    for t = 1:horizon
        τ_buf[t] = ϕ(s)
        a = action(policy, b)
        sp, o = @gen(:sp, :o)(p, s, a, x[:, t])
        bp = update(up, b, a , o)
        s = sp
        b = bp
    end
    τ = copy(τ_buf)
    d = minimum(τ)
    Turing.@addlogprob! logpdf(Normal(0.0, ϵ), d)
end



