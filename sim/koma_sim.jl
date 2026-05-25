# Import the package
using KomaMRI
using IterTools
using MAT
using FFTW
using GLMakie
using Statistics

using FileIO
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
include("../utils/wrap.jl")
include("../kernels/f_kernel.jl")

function simular_fieldmap(fieldmap_mT, B_0REF, reconstruir, output_dir, experiment_name, low, high)
    # Define scanner, object and sequence
    sys = Scanner(B0=0.04515, B1= 0.5e-10, Gmax = 0.00002, Smax= 500, ADC_Δt= 2.0e-6, seq_Δt= 1.0e-5, GR_Δt= 1.0e-5, RF_Δt= 1.0e-6,RF_ring_down_T= 2.0e-5, RF_dead_time_T= 0.0001,ADC_dead_time_T= 1.0e-5)
    a = Float64.(-0.075:0.15/128:0.075-0.15/128)
    x = repeat(a, 128)
    b = repeat(a', 128)
    y = reduce(vcat, b)

    file_lectura = matopen("sim/objeto.mat")
    objeto = read(file_lectura,"Ib")
    threshold = 2*10^-7          
    mask = objeto .> threshold  
    
    ρ = Float64.(reduce(vcat, Float32.(mask) )) 

    fieldmap_3D = fieldmap_mT
    sz_z = size(fieldmap_3D)[3]
    iz_center = round(Int, (sz_z + 1) / 2)
    fieldmap_2D = fieldmap_3D[:, :, iz_center]
    
    ΔB = B_0REF.*ones(size(vec(fieldmap_2D))) - vec(fieldmap_2D)

    # Definir los demás parámetros del Phantom
    T1 = 0.5*ones(128*128)
    T2 = 0.2*ones(128*128)
    T2s = 0.05*ones(128*128)
    
    γ_rad_mT = 2π * 42.57e6 * 1e-3 # rad/(s*mT)
    Δw = -γ_rad_mT * ΔB

    obj = Phantom(name="miss phantom", x=x, y=y, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw)

    # Simulación
    sim_params = KomaMRICore.default_sim_params()
    sim_params["return_type"] = "mat"
    seq = read_seq("sim/seq/se_BOOST2.seq")
    raw = simulate(obj, seq, sys; sim_params)

    # Automatically create the experiment directory and an images subdirectory
    img_dir = joinpath(output_dir, "imgs")
    mkpath(img_dir)

    # Save the .mat file inside the contained folder
    name_file = joinpath(output_dir, "k_Shim_OR_Campo.mat") 
    matwrite(name_file, Dict("raw" => raw))
    
    if reconstruir == 1
        data = reshape(raw, 128, 128)
        imagen = rotl90(abs.(fftshift(ifft(data))))
        fig = Figure()  
        ax  = Axis(fig[1, 1],
                aspect = DataAspect(),
                yreversed = true,
                title = string("Reconstruction simulation: ", experiment_name))

        hm = heatmap!(ax, imagen; colormap = :grays, colorrange = (low, high))
        Colorbar(fig[1, 2], hm)
        
        # Save image in the dynamic subfolder
        img_path = joinpath(img_dir, "imagen_simulada_$(experiment_name).png")
        save(img_path, fig)
        display(fig) 
    else
        println("Simulación terminada para $experiment_name")
    end
    return raw
end



include("../setup.jl")

# CONFIGURATION
experiment_name = "SH_oneradius_PRESHIM" #"SH_oneradius_run170"
file_input      = "" #"runs/SH_JOSH_oneradius/best_res_170.jld2" #Si es preshim se usa el fieldma cargado en setup
shimmed         = false
reconstruir     = 1
low, high       = 0, 1


let name = experiment_name, file = file_input

    # Saving dir
    output_dir = joinpath("sim", "sim_runs", name)
    if shimmed
        # Load JLD2
        data_jld2 = JLD2.load(file)
        λ = data_jld2["λ"]
        bestθ = CuArray(data_jld2["bestθ"])
        final_state = CuArray(data_jld2["final_state"])

        # Compute and references
        by_mean = CuArray([0.0f0])
        RMS_operation(bestθ, λ, final_state)
        @cuda threads=threads blocks=blocks shmem=shmem_sum _mean!(B, by_mean, N, Nmsk)

        B_0REF = Array(by_mean)[1]
        fieldmap_sim = Array(B)

        # Sim
        raw_1 = simular_fieldmap(fieldmap_sim, B_0REF, reconstruir, output_dir, name, low, high)
    else
        B_0REF = mean(fieldmap)
        fieldmap_sim = fieldmap
        raw_1 = simular_fieldmap(fieldmap_sim, B_0REF, reconstruir, output_dir, name, low, high)
    end
end