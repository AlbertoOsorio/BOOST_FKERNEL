### BOOST es un script que recibe un fieldmap en mT y entrega un fieldmap en mT

include("../kernels/f_kernel.jl")
include("../utils/grid_utils.jl")
include("../utils/ppms.jl")

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
const λ  = 0.5            # peso RMS(∂B/∂*) en mT/m (solo cascarón)en nuestra funcion objetivo este es el λ
const w = 1.0            # peso rango (max-min)/mean en mT (solo cascarón) en nuestra funcion objetivo esto es 1
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

P_cpu = hcat(posiciones...)             # Convierte a matrix 3x336


fld = Float32.(CuArray(fieldmap))
msk = CuArray(dmask)

masked_count(dmask::CuArray{TM,3}) where {TM<:AbstractFloat} = Float64(CUDA.sum(dmask))
Nmsk   = Float32(masked_count(msk))

xg_mm, yg_mm, zg_mm = build_axes_mm(dims, resmm)
grid = make_grid_gpu_from_axes_mm(xg_mm, yg_mm, zg_mm)

P = Float32.(CuArray(P_cpu .* 0.001))

M = similar(P)
B = similar(grid.X)

by_mean = CuArray([0.0f0])
stdiv = CuArray([0.0f0])

threads = 512 
N = Int32(length(grid.X))
blocks = cld(N, threads) 


shmem_sum = threads * sizeof(Float32) 

Θ = Float32.(CuArray(θ0))
μ = Float32.(CuArray(μ_base))
m = Int32(size(P_cpu, 2))


function mintest()

    @cuda threads=threads blocks=blocks                  _M!(Θ, μ, m, M)

    @cuda threads=threads blocks=blocks                  _Btotmasked!(fld, B,                                              # Valores base y alocaciones
                                                                grid.X, grid.Y, grid.Z,     # Grid de evaluación
                                                                P, M, m,                                               # Pos y θ de vec momento dipolo
                                                                msk, N)                                                   # Máscara

    @cuda threads=threads blocks=blocks shmem=shmem_sum  _mean!(B, by_mean, N, Nmsk)

    @cuda threads=threads blocks=blocks shmem=shmem_sum  _std!(B, by_mean, stdiv, N, Nmsk)

end
