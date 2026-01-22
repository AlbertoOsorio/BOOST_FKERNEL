### BOOST es un script que recibe un fieldmap en mT y entrega un fieldmap en mT
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie



include("../utils/grid_utils.jl")
include("../utils/ppm_copy.jl")
include("../utils/imanes.jl")

const BATCH_M = 64
const FILE = "data/By_SH.jld2"                      # Ajusta si cambiaste el nombre
@load FILE By_grid xg yg zg modelBy x y z By   # Todo en mT y mm
fieldmap = By_grid


B0y  = fieldmap                      # mT, Array{Float64,3}
const dB0y = CUDA.has_cuda() ? CuArray(Float32.(B0y)) : nothing

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
println("M (posiciones) cartesianas = ", M)
println(posiciones)
pos = posiciones
# Convertimos el Vector{NTuple{3,Float64}} a tres vectores x,y,z
xs = [p[1] for p in pos]
ys = [p[2] for p in pos]
zs = [p[3] for p in pos]

# 2) Definimos el cilindro interior (diámetro 200 mm → radio 100 mm)
R_inner = 200.0  # mm

zmin = minimum(zs)
zmax = maximum(zs)

# Opcional: un pequeño margen para que el cilindro sobresalga un poco
margin = 10.0
z0 = zmin - margin
z1 = zmax + margin


# =======================
# GA setup (límites y θ inicial)
# =======================
lower = fill(0.0,   M)   # grados
upper = fill(180.0, M)
θ0    = 150.0 .* ones(M)

# Discretización a 5°
disc5(θ) = [5.0 * clamp(round(Int, d/5), 0, 36) for d in θ]  # múltiplos de 5°

# =======================
# Kernels GPU: ∂/∂y, ∂/∂x, ∂/∂z
# =======================
@inline function _idx(i,j,k, nx,ny,nz)
    return i + (j-1)*nx + (k-1)*nx*ny
end

function _grad_y_kernel!(Gy, By, dy, nx::Int32, ny::Int32, nz::Int32)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    N = nx*ny*nz
    if i > N; return; end
    tmp  = i - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    ii   = (tmp2 % nx) + 1

    if 1 < j < ny
        Gy[i] = (By[_idx(ii,j+1,k,nx,ny,nz)] - By[_idx(ii,j-1,k,nx,ny,nz)]) / (2f0*dy)
    elseif j == 1
        Gy[i] = (By[_idx(ii,2,k,nx,ny,nz)]   - By[_idx(ii,1,k,nx,ny,nz)])   / dy
    else
        Gy[i] = (By[_idx(ii,ny,k,nx,ny,nz)]  - By[_idx(ii,ny-1,k,nx,ny,nz)]) / dy
    end
    return
end

function _grad_y_cuda!(Gy::CuArray{T,3}, By::CuArray{T,3}, dy::T) where {T<:AbstractFloat}
    nx,ny,nz = size(By)
    @cuda threads=256 blocks=cld(length(By),256) _grad_y_kernel!(Gy, By, dy, Int32(nx),Int32(ny),Int32(nz))
    CUDA.synchronize()
    return Gy
end

function _grad_x_kernel!(Gx, B, dx, nx::Int32, ny::Int32, nz::Int32)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    N = nx*ny*nz
    if i > N; return; end

    tmp  = i - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    ii   = (tmp2 % nx) + 1

    if 1 < ii < nx
        Gx[i] = (B[_idx(ii+1,j,k,nx,ny,nz)] - B[_idx(ii-1,j,k,nx,ny,nz)]) / (2f0*dx)
    elseif ii == 1
        Gx[i] = (B[_idx(2,j,k,nx,ny,nz)] - B[_idx(1,j,k,nx,ny,nz)]) / dx
    else
        Gx[i] = (B[_idx(nx,j,k,nx,ny,nz)] - B[_idx(nx-1,j,k,nx,ny,nz)]) / dx
    end
    return
end

function _grad_z_kernel!(Gz, B, dz, nx::Int32, ny::Int32, nz::Int32)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    N = nx*ny*nz
    if i > N; return; end

    tmp  = i - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    ii   = (tmp2 % nx) + 1

    if 1 < k < nz
        Gz[i] = (B[_idx(ii,j,k+1,nx,ny,nz)] - B[_idx(ii,j,k-1,nx,ny,nz)]) / (2f0*dz)
    elseif k == 1
        Gz[i] = (B[_idx(ii,j,2,nx,ny,nz)] - B[_idx(ii,j,1,nx,ny,nz)]) / dz
    else
        Gz[i] = (B[_idx(ii,j,nz,nx,ny,nz)] - B[_idx(ii,j,nz-1,nx,ny,nz)]) / dz
    end
    return
end

function _grad_x_cuda!(Gx::CuArray{T,3}, B::CuArray{T,3}, dx::T) where {T<:AbstractFloat}
    nx,ny,nz = size(B)
    @cuda threads=256 blocks=cld(length(B),256) _grad_x_kernel!(Gx, B, dx, Int32(nx), Int32(ny), Int32(nz))
    CUDA.synchronize()
    return Gx
end

function _grad_z_cuda!(Gz::CuArray{T,3}, B::CuArray{T,3}, dz::T) where {T<:AbstractFloat}
    nx, ny, nz = size(B)
    @cuda threads=256 blocks=cld(length(B),256) _grad_z_kernel!(Gz, B, dz, Int32(nx), Int32(ny), Int32(nz))
    CUDA.synchronize()
    return Gz
end


# =======================
# Objetivo combinado: gradiente + rango (max-min)
# =======================

# ===== Helpers de reducciones enmascaradas (GPU) - genéricos =====
# A y mask pueden tener distintos eltypes; casteamos la máscara a eltype(A).
function masked_minmax(dA::CuArray{TA,3}, dmask::CuArray{TM,3}) where {TA<:AbstractFloat, TM<:AbstractFloat}
    dmaskA = (TA === TM) ? dmask : CUDA.map(TA, dmask)
    A_for_min = ifelse.(dmaskA .> zero(TA), dA, convert(TA,  Inf))
    A_for_max = ifelse.(dmaskA .> zero(TA), dA, convert(TA, -Inf))
    return CUDA.minimum(A_for_min), CUDA.maximum(A_for_max)
end

# Cuenta voxels "activos" del cascarón (independiente del tipo)
masked_count(dmask::CuArray{TM,3}) where {TM<:AbstractFloat} = Float64(CUDA.sum(dmask))

# --- 100% GPU --- (SOLO cascarón)

dmask32  = Float32.(dmask .> 0)                      # asegúrate de 0/1 Float32
dmask    = CuArray(dmask32)

function objective_gpu_allgpu_ranged( θ )
    #θdeg = DISC_5 ? disc5(θdeg_raw) : θdeg_raw

    # Campo de imanes (mT) en GPU
    dBy = campo_imanes_gpu(posiciones, θ, dims, resmm;
                           center_grid=true, center_mm=(0,0,0),
                           axis=:x, B1cm_T=B1CM_T, to_mT=true)

    # --- asegurar eltype=Float32 en GPU ---
    if eltype(dBy) !== Float32
        dBy = CUDA.map(Float32, dBy)
    end

    # Campo total (ambos Float32)
    dBtot = dB0y .+ dBy

    # Gradientes (mT/m) en GPU
    dGy = similar(dBtot); _grad_y_cuda!(dGy, dBtot, dy_m)
    dGx = similar(dBtot); _grad_x_cuda!(dGx, dBtot, dx_m)
    dGz = similar(dBtot); _grad_z_cuda!(dGz, dBtot, dz_m)

    # Enmascarar gradientes para RMS solo en el cascarón
    dGy .*= dmask
    dGx .*= dmask
    dGz .*= dmask

    Nmask   = masked_count(dmask)
    s       = CUDA.sum(abs2, dGy) + CUDA.sum(abs2, dGx)+ CUDA.sum(abs2, dGz)
    grad_rms = sqrt(Float64(s) / Float64(Nmask))          # mT/m

    # Rango del campo SOLO en el cascarón
    by_min, by_max = masked_minmax(dBtot, dmask)
    range_by = Float64(by_max - by_min)               # mT

    return w_range * range_by + w_grad * grad_rms
end



function clamp!(v, lo, hi)
    @inbounds for i in eachindex(v)
        v[i] = min(max(v[i], lo[i]), hi[i])
    end
    return v
end

# Mutación gaussiana con radio (step_deg), discretizada a 5°
function mutate!(θ, step_deg, lo, hi)
    @inbounds for i in eachindex(θ)
        if rand() < 0.3               # 30% de genes mutan
            θ[i] += step_deg * randn()
        end
    end
    clamp!(θ, lo, hi)
    return θ
end

# Enfriamiento exponencial
cool(t0, alpha, k) = t0 * (alpha^k)

# Bucle principal de SA con reinicios
function optimize_SA(f; θ_init=θ0, lower=lower, upper=upper,
                     iters=10, restarts=5,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)

    bestθ_global = copy(θ_init)
    bestf_global = f(bestθ_global)
    dBy = campo_imanes_gpu(posiciones, bestθ_global, dims, resmm; axis=:y, B1cm_T=B1CM_T, to_mT=true)
    bestcampo_global = dB0y .+ dBy

    for r in 1:restarts
        θ = (r==1 ? copy(bestθ_global) : lower .+ rand(length(θ_init)) .* (upper .- lower))

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

end

f_eval = θ -> objective_gpu_allgpu_ranged(θ)