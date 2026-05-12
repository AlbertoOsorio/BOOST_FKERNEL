using GLMakie
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT


include("utils/wrap.jl")
include("utils/ppm_report.jl")
include("setup.jl")


function get_DR()
    D = []
    R = []
    for i = 1:100 
        include("setup.jl")
        f_data = "runs/L_curve_150_params_T100999/lambdaeq_$(i).jld2"
        @load f_data  λ bestθ final_state ppm

        by_mean = CuArray([0.0f0])

        RMS_operation(bestθ, λ, final_state)

        push!(R, Array(grad_rms))

        push!(D, Array((by_max - by_min) / by_mean))

    end
    return D, R
end

function plot_L_curve(D_vals::AbstractVector, R_vals::AbstractVector)
    @assert length(D_vals) == length(R_vals) "D y R deben tener el mismo largo"
    @assert length(D_vals) == 100 "Se esperan 100 samples"

    # Construcción de lambdas (solo para referencia/coloreo si quieres)
    lambdas = range(0.01, 1.0, length=100)

    # Validación básica (evitar log de valores no positivos)
    @assert all(D_vals .> 0) "Todos los D(x_λ) deben ser > 0"
    @assert all(R_vals .> 0) "Todos los R(x_λ) deben ser > 0"

    # Transformación log-log
    logD = log10.(D_vals)
    logR = log10.(R_vals)

    # Figura
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "log10(D(x_λ))",
        ylabel = "log10(R(x_λ))",
        title  = "Curva L"
    )

    # Curva
    lines!(ax, logD, logR, linewidth = 2)
    scatter!(ax, logD, logR)

    # (Opcional) marcar extremos
    text!(ax, logD[1], logR[1], text = "λ=$(round(lambdas[1], digits=3))", align = (:left, :bottom))
    text!(ax, logD[end], logR[end], text = "λ=$(round(lambdas[end], digits=3))", align = (:right, :top))

    fig
end