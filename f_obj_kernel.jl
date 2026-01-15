using CUDA
using LinearAlgebra

const to_rad_f32 = 1.74532925f-2 # π/180 precalculado como Float32
const scale = 1.0f-7 # μ0 / 4π

function _M!(θ, μ, m, M)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx > m; return; end
    @inbounds begin
        θin = θ[idx] * to_rad_f32
        μin = μ[idx]
        M[1, idx] = cos(θin) * μin
        M[2, idx] = sin(θin) * μin
        M[3, idx] = 0.0
    end
    return
end



function _Btot!(fieldmap, B,                                                   # Valores base y alocaciones
                X, Y, Z,                                                       # Grid de evaluación
                P, M, m, N)                                                 # Pos y θ de vec momento dipolo

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x                                   

    _By_from_shim(X, Y, Z, P, M, B, m, N, idx)
    
    if idx <= N
        B[idx] += fieldmap[idx] # Bytot
    end
    return 
end

@inline function _By_from_shim(X, Y, Z,  # X, Y y Z Son grillas de tamaño nx x ny x nz, las que contienen valores escalares. 
                                  # En vez de entregar los vectores posicion entregamos 3 campos escalares en grilla con x y z cada uno
                            P, M, B, m,
                            N, idx)
    
    # Inicializamos acumulador de By
    By = 0.0f0
    
    # Cargamos punto de eval
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    @inbounds if idx <= N
        Rx = X[idx]
        Ry = Y[idx]
        Rz = Z[idx]
    end

    if idx <= N
        @inbounds for k in 1:m
            # Obtener dipolo desde memoria compartida
            μx, μy, μz = M[1, k], M[2, k], M[3, k]
            px, py, pz = P[1, k], P[2, k], P[3, k]
            
            dx = Rx - px
            dy = Ry - py
            dz = Rz - pz
            
            r2 = dx*dx + dy*dy + dz*dz
            r = sqrt(r2)
            
            if r > 1.0f-9 # Evitamos div por 0
                inv_r3 = 1.0f0 / (r2 * r)
                inv_r5 = inv_r3 / r2
                dot_mr = dx*μx + dy*μy + dz*μz

                By += 3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3
            end
        end
    end
        
    # Write final result to Global Memory
    if idx <= N
        @inbounds B[idx] = By * scale * 1000 # mT
    end
    return
end

# Determina el indice 1D de acceso al array
@inline function _idx(i,j,k, nx,ny,nz)
    return i + (j-1)*nx + (k-1)*nx*ny
end

function _grad!(B, Gx, Gy, Gz, d, nx, ny, nz, N)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    tmp  = idx - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    i   = (tmp2 % nx) + 1

    if idx > N; return; end
    @inbounds begin
        #grad x
        if 1 < i < nx
            Gx[idx] = (B[_idx(i+1,j,k,nx,ny,nz)] - B[_idx(i-1,j,k,nx,ny,nz)]) / (2f0*d)
        elseif i == 1
            Gx[idx] = (B[_idx(2,j,k,nx,ny,nz)] - B[_idx(1,j,k,nx,ny,nz)]) / d
        else
            Gx[idx] = (B[_idx(nx,j,k,nx,ny,nz)] - B[_idx(nx-1,j,k,nx,ny,nz)]) / d
        end

        #grad y
        if 1 < j < ny
            Gy[idx] = (B[_idx(i,j+1,k,nx,ny,nz)] - B[_idx(i,j-1,k,nx,ny,nz)]) / (2f0*d)
        elseif j == 1
            Gy[idx] = (B[_idx(i,2,k,nx,ny,nz)]   - B[_idx(i,1,k,nx,ny,nz)])   / d
        else
            Gy[idx] = (B[_idx(i,ny,k,nx,ny,nz)]  - B[_idx(i,ny-1,k,nx,ny,nz)]) / d
        end

        #grad z
        if 1 < k < nz
            Gz[idx] = (B[_idx(i,j,k+1,nx,ny,nz)] - B[_idx(i,j,k-1,nx,ny,nz)]) / (2f0*d)
        elseif k == 1
            Gz[idx] = (B[_idx(i,j,2,nx,ny,nz)] - B[_idx(i,j,1,nx,ny,nz)]) / d
        else
            Gz[idx] = (B[_idx(i,j,nz,nx,ny,nz)] - B[_idx(i,j,nz-1,nx,ny,nz)]) / d
        end
    end
    sync_threads()

    return 
end


function _metrics!(B, by_min, by_max, grad_rms, Gx, Gy, Gz, mask, N, Nmask)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_min = CuDynamicSharedArray(eltype(B), blockDim().x)                                        # Para calcular min en Btot masked
    shared_max = CuDynamicSharedArray(eltype(B), blockDim().x, blockDim().x * sizeof(eltype(B)))      # Para calcular max en Btot masked
    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x, 2 * blockDim().x * sizeof(eltype(B)))  # Para calcular RMS dBytot

    # Inicializamos shmem con un valor
    shared_min[tid] = typemax(eltype(B))
    shared_max[tid] = typemin(eltype(B))
    shared_sum[tid] = zero(eltype(B))

    if idx == 1
        by_min[idx] = typemax(eltype(B))
        by_max[idx] = typemin(eltype(B))
        grad_rms[idx] = 0.0f0
    end

    # Cargamos un punto y adicionalmente guardamos val²
    if idx <= N
        B[idx]  *= mask[idx]
        Gx[idx] *= mask[idx]
        Gy[idx] *= mask[idx]
        Gz[idx] *= mask[idx]
        sync_threads()

        val = B[idx]
        
        shared_min[tid] = val
        shared_max[tid] = val
        shared_sum[tid] = (Gx[idx]^2 + Gy[idx]^2 + Gz[idx]^2)/Nmask
    end
   
    # Aseguramos que toda la memoria termino de ser cargada
    sync_threads()

    # Tree Reduction
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            if shared_min[tid] == 0.0
                shared_min[tid] = shared_min[tid + s]
            elseif shared_min[tid + s] == 0.0
            else
                shared_min[tid] = min(shared_min[tid], shared_min[tid + s])
            end
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s])
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Updates to Global Memory
    if tid == 1
        CUDA.@atomic by_min[] = min(by_min[], shared_min[1])
        CUDA.@atomic by_max[] = max(by_max[], shared_max[1])
        CUDA.@atomic grad_rms[] += shared_sum[1]
    end
    return
end

function f_val!(by_min, by_max, grad_rms, coef)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx == 1
        coef[idx] = w * (by_max[idx] - by_min[idx]) + λ * sqrt(grad_rms[idx])
    end
    return
end