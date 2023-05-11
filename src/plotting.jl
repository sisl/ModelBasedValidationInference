using Plots

function plot_failures(p, failure_events, plot_fun; fig=plot(), kwargs...)
    fail_hists = histories(failure_events)
    lps = log_probs(failure_events)
    lps .+= abs(minimum(lps))
    ascale = (maximum(lps) - minimum(lps))

    for (i,h) in enumerate(fail_hists)
        alpha = 1 - (maximum(lps) - lps[i])/ascale
        plot_fun(p, h; fig=fig, alpha=alpha, kwargs...)
    end
    return fig
end

rectangle(w, h, x, y) = Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
