### BOOST es un script que recibe un fieldmap en mT y entrega un fieldmap en mT

include("std_kernel.jl")
include("utils/grid_utils.jl")
include("utils/ppms.jl")

using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie

const BATCH_M = 64
const FILE = "data/By_SH.jld2"                      # Ajusta si cambiaste el nombre
@load FILE By_grid xg yg zg modelBy x y z By   # Todo en mT y mm
fieldmap = By_grid

## Definir el tamaño del cascaron en el que mediremos los errores
Rmin = 0.00   # mm
Rmax = 100.0  # mm


## Definir los anillos en las bandejas en los que pondremos imanes para hacer shimming mas los que están usados
positions_in_tray_occupied   = Int[]                 # los que ya están
positions_in_tray_new_wished = [-14, -6, 6, 14]      # los que queremos ocupar
title_1 = "posiciones_imanes_shimming"               # nombre de la figura


## Ahora defino constantes
const B1CM_T = 0.012     # campo de cada iman a 1cm
const DISC_5 = true      # Las rotaciones de cada iman solo pueden ser un cm
w_grad  = 0.5            # peso RMS(∂B/∂*) en mT/m (solo cascarón)en nuestra funcion objetivo este es el λ
w_range = 1.0            # peso rango (max-min)/mean en mT (solo cascarón) en nuestra funcion objetivo esto es 1
iteraciones = 10
restarts_1 = 2

#Definimos el step de la grilla del fieldmap. Viene del jld2
dx = length(xg) > 1 ? minimum(abs.(diff(xg))) : 0.0
dy = length(yg) > 1 ? minimum(abs.(diff(yg))) : 0.0
dz = length(zg) > 1 ? minimum(abs.(diff(zg))) : 0.0
resmm = (dx, dy, dz)  

cx, cy, cz = modelBy.center

# radios en cada voxel (CPU)
Rx = reshape(xg .- cx, :, 1, 1)
Ry = reshape(yg .- cy, 1, :, 1)
Rz = reshape(zg .- cz, 1, 1, :)
rgrid = sqrt.(Rx.^2 .+ Ry.^2 .+ Rz.^2)

# Definimos dmask
Δ  = max(dx, dy, dz) 
tol  = 1e-3 * Δ 
mask_shell_bool = (rgrid .>= (Rmin - tol)) .& (rgrid .<= (Rmax + tol))
dmask    = Float32.(mask_shell_bool)                


# resoluciones en metros
dx_m = Float32(dx * 1e-3)
dy_m = Float32(dy * 1e-3)
dz_m = Float32(dz * 1e-3)
dims = size(fieldmap)

posiciones = positions_from_rings_mm(positions_in_tray_new_wished;
    occupied_trays         = positions_in_tray_occupied,
    shim_radius_mm         = 235.0,
    mags_per_segment       = 7,
    num_segments           = 12,
    angle_per_segment_deg  = 2*(180 - 169.68),
    angular_offset_deg     = 0.0)
M = length(posiciones)

lower = fill(0.0,   M)                  # grados
upper = fill(180.0, M)
θ0    = 150.0 .* ones(M)
μ_base = 0.06 .* ones(M)                # TODO Determinar valor real de la magnitud de los imanes de shimming

M0_cpu = transpose(hcat(θ0, μ_base))
P_cpu = hcat(posiciones...)             # Convierte a matrix 3x336

# =======================
# Optimización: Simulated Annealing + reinicios
# =======================

# Discretiza a múltiplos de 5° en [0,180]
quant5!(θ) = (θ .= [5.0 * clamp(round(Int, d/5), 0, 36) for d in θ])

function clamp!(v, lo, hi)
    @inbounds for i in eachindex(v)
        v[i] = min(max(v[i], lo[i]), hi[i])
    end
    return v
end


## TODO agregar a gpu
# Mutación gaussiana con radio (step_deg), discretizada a 5°
function mutate!(θ, step_deg, lo, hi)
    @inbounds for i in eachindex(θ)
        if rand() < 0.3               # 30% de genes mutan
            θ[i] += step_deg * randn()
        end
    end
    clamp!(θ, lo, hi)
    quant5!(θ)
    return θ
end

# Enfriamiento exponencial
cool(t0, alpha, k) = t0 * (alpha^k)

# Bucle principal de SA con reinicios
function optimize_SA(f; θ_init=θ0, lower=lower, upper=upper,
                     iters=2_000, restarts=5,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)

    bestθ_global = copy(θ_init); quant5!(bestθ_global)
    bestf_global = f(bestθ_global)

    for r in 1:restarts
        θ = (r==1 ? copy(bestθ_global) : lower .+ rand(length(θ_init)) .* (upper .- lower))
        quant5!(θ)
        fθ = f(θ)

        step = step0
        T = T0
        step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

        for k in 1:iters
            θnew = copy(θ)
            mutate!(θnew, step, lower, upper)
            fnew = f(θnew)

            if (fnew < fθ) || (rand() < exp(-(fnew - fθ)/max(T,1e-9)))
                θ .= θnew
                fθ = fnew
                if fθ < bestf_global
                    bestf_global = fθ
                    bestθ_global .= θ
                end
            end

            T = cool(T0, alpha, k)
            step = max(step*step_decay, step_min)

            if (k % report_every == 0)
                @info "SA restart=$r iter=$k  f=$fθ  best=$bestf_global  T=$(round(T,digits=4))  step=$(round(step,digits=2))"
            end
        end
    end

    return bestf_global, bestθ_global
end


fld = CuArray(fieldmap)
msk = CuArray(dmask)

masked_count(dmask::CuArray{TM,3}) where {TM<:AbstractFloat} = Float64(CUDA.sum(dmask))
Nmsk   = Float32(masked_count(msk))

xg_mm, yg_mm, zg_mm = build_axes_mm(dims, resmm)
grid = make_grid_gpu_from_axes_mm(xg_mm, yg_mm, zg_mm)

P = CuArray(P_cpu .* 0.001)
B = similar(grid.X)

by_mean = CuArray([0.0f0])
stdiv = CuArray([0.0f0])


threads = 256 
N = length(grid.X)
blocks = cld(N, threads) 
shmem =  5 * BATCH_M * sizeof(Float32)
shmem_sum = threads * sizeof(Float32)




function mintest()
    M = CuArray(M0_cpu)
    m = Int32(size(M0_cpu, 2))
    @cuda threads=threads blocks=blocks shmem=shmem  _Btotmasked!(fld, B,                                              # Valores base y alocaciones
                                                                grid.X, grid.Y, grid.Z, grid.nx, grid.ny, grid.nz,     # Grid de evaluación
                                                                P, M, m,                                               # Pos y θ de vec momento dipolo
                                                                msk)                                                   # Máscara

    @cuda threads=threads blocks=blocks shmem=shmem_sum  _mean!(B, by_mean, N, Nmsk)

    @cuda threads=threads blocks=blocks shmem=shmem_sum  _std!(B, by_mean,stdiv, N, Nmsk)

end

# ---- enlaza con tu objetivo GPU ya definido arriba ----
#f_eval = θ -> objective_gpu_allgpu_ranged(θ)



#bestf, bestθ, bestcampo_global = optimize_SA(f_eval; θ_init=θ0, iters=iteraciones, restarts=restarts_1,
#                           T0=0.05, alpha=0.995, step0=10.0, step_min=1.0,
#                           report_every=100)


#println("Optimizacion terminada")
#println("fmin = ", bestf)
#println("rotaciones (deg) = ", (DISC_5 ? disc5(bestθ) : bestθ)[1:min(10,end)])
