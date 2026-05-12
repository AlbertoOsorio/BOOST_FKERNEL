using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie

function save_ppm()
    ppms_saved = []
    for baseline = 1:100
        FILE = "runs/L_curve_150_params_T100999/lambdaeq_$(baseline).jld2"
        @load FILE  λ bestθ final_state ppm
        push!(ppms_saved, ppm)
    end
    return ppms_saved
end

function generate_ppm_graph()
    # Tus datos (ejemplo)
    final_ppms = save_ppm()

    y = vec(Array(reduce(vcat, final_ppms)))

    # Construir eje x (asumiendo que partes en 0)
    x = 0:0.1:0.1*(length(final_ppms)-1)

    # Crear figura
    fig = Figure()
    ax = Axis(fig[1, 1];
        title = "PPM vs λ",
        xlabel = "λ value",
        ylabel = "PPM"
    )

    # Graficar puntos + líneas
    lines!(ax, x, y)       # conecta con líneas
    scatter!(ax, x, y)     # dibuja los puntos

    fig
    save(string("runs/L_curve_150_params_T100999/","PPMvslambda.png"), fig)

end

a = [0.85, 0.87, 0.89, 0.9, 0.92, 0.95, 0.96, 0.97, 0.99, 0.999]


const FILE = "data/parallelepiped_150x150x150.jld2"
@load FILE  By_3D xg yg zg
fieldmap = By_3D

# Assuming your mask array is named 'mask_3D'
for i = 1:11
    slice = By_3D[:,:,i]
    mask_slice = dmask[:,:,i]

    # Create a copy for plotting and apply the mask
    # We replace 0s (or whatever your 'masked' value is) with NaN
    plot_data = copy(slice)
    plot_data[mask_slice .== 0] .= NaN 

    fig = Figure(resolution = (800, 600))

    ax = Axis(fig[1, 1];
        title = "Fieldmap slice z = $(i)",
        xlabel = "x mm",
        ylabel = "y mm"
    )

    # Use nan_color to make the masked areas black
    hm = heatmap!(ax, xg, yg, plot_data; 
        colormap = :viridis,
        colorrange = (46.21437, 47.50534),
        nan_color = :black  # This turns the NaNs black
    )
    
    Colorbar(fig[1, 2], hm, label = "mT")
    
    dir = "imgs/150baseline_msk/"
    mkpath(dir) 
    save(joinpath(dir, "$(i)_slice.png"), fig)
end