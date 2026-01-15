using CUDA
using LinearAlgebra


function _Btotmasked!(fieldmap, B,                           # Valores base y alocaciones
                X, Y, Z, nx, ny, nz,                                        # Grid de evaluación
                P, M, m,                                                    # Pos y θ de vec momento dipolo
                mask)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Construimos los indices de la grilla para acceder 1D al arreglo CuArray{T, 3}
    N = nx*ny*nz                # Cantidad de puntos de evaluación

    tmp  = idx - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    i   = (tmp2 % nx) + 1

    shmem      = CuStaticSharedArray(Float32, (5, BATCH_M))                                      

    _By_from_shim(X, Y, Z, P, M, B, m, shmem, N, idx)

    if idx <= N
        B[idx] += fieldmap[idx] # Bytot
        B[idx] *= mask[idx]    # Aplicamos mascara
    end

    return # Es un kernel mutable, no debemos retornar nada
end

@inline function _By_from_shim(X, Y, Z,  # X, Y y Z Son grillas de tamaño nx x ny x nz, las que contienen valores escalares. 
                                  # En vez de entregar los vectores posicion entregamos 3 campos escalares en grilla con x y z cada uno
                            P, M, B, m, shmem,
                            N, idx)
    
    # Inicializamos acumulador de By
    By = 0.0f0
    
    # Cargamos punto de eval
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    if idx <= N
        Rx = X[idx]
        Ry = Y[idx]
        Rz = Z[idx]
    end

    jb = 1
    while jb <= m
        batch_size = min(BATCH_M, m - jb + 1)
        
        # 3. MASKED COLLABORATIVE LOAD
        # We need to fill shmem[:, 1:batch_size]. 
        # Total elements to load = 5 * batch_size.
        # We use all threads in the block to load these values in parallel.
        total_elements = 5 * batch_size
        tid = threadIdx().x
        while tid <= total_elements
            # Map linear tid to (row, col) in shared memory
            # col: which dipole in the batch
            # row: which component (1-3: M, 4-6: P)
            col = (tid - 1) ÷ 5 + 1
            row = (tid - 1) % 5 + 1
            
            global_col = jb + col - 1
            
            if row <= 2
                shmem[row, col] = M[row, global_col]
            else
                shmem[row, col] = P[row - 2, global_col]
            end
            tid += blockDim().x
        end
        
        # Aseguramos que todos los datos finalizaron de ser cargados a shmem antes de comenzar los calculos
        sync_threads()

        # Iteramos sobre cada dipolo del batch en el que nos encontramos
        if idx <= N
            for l in 1:batch_size
                # Obtener dipolo desde memoria compartida
                θ, μ = shmem[1, l], shmem[2, l]
                μx = cos(θ)*μ; μy = sin(θ)*μ; μz = 0f0

                px, py, pz = shmem[3, l], shmem[4, l], shmem[5, l]
                
                dx = Rx - px
                dy = Ry - py
                dz = Rz - pz
                
                r2 = dx*dx + dy*dy + dz*dz
                r = sqrt(r2)
                
                if r > 1.0f-9 # Evitamos div por 0
                    inv_r3 = 1.0f0 / (r2 * r)
                    inv_r5 = inv_r3 / r2
                    dot_mr = dx*μx + dy*μy + dz*μz
                    scale = 1.0f-7 # μ0 / 4π

                    By += scale * (3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3)
                end
            end
        end
        
        # Aseguramos que todos los threads terminen antes de cargar la siguiente batch
        sync_threads()
        jb += BATCH_M
    end

    # Write final result to Global Memory
    if idx <= N
        B[idx] = By * 1000 # mT
    end
    return
end


function _mean!(B, by_mean, N, Nmask)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x)

    # Inicializamos shmem con un valor
    shared_sum[tid] = zero(eltype(B))

    # Cargamos un punto y adicionalmente guardamos val²
    if idx <= N
        val = B[idx]
        shared_sum[tid] = val / Nmask
    end
   
    # Aseguramos que toda la memoria termino de ser cargada
    sync_threads()

    # Tree Reduction
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Updates to Global Memory
    if tid == 1
        CUDA.@atomic by_mean[] += shared_sum[1]
    end
    
    return
end

function _std!(B, by_mean,stdiv, N, Nmask)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x)

    # Inicializamos shmem con un valor
    shared_sum[tid] = zero(eltype(B))

    # Cargamos un punto y adicionalmente guardamos val²
    if idx <= N
        val = B[idx]
        if val == 0.0
            shared_sum[tid] = 0.0
        else
            μ = by_mean[1]
            shared_sum[tid] = ((val - μ)^2)/Nmask
        end
    end
   
    # Aseguramos que toda la memoria termino de ser cargada
    sync_threads()

    # Tree Reduction
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Updates to Global Memory
    if tid == 1
        CUDA.@atomic stdiv[] += shared_sum[1]
    end

    return
end