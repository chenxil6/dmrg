# main.jl

using ITensors

println("== CPU quick test ==")
i = Index(2, "i")
j = Index(2, "j")

A = random_itensor(i, j)   # (newer name; randomITensor still works)
B = dag(A)
C = A * B

@show size(A)
@show hasinds(C, prime(i), i)
println("CPU ITensors is working ✅\n")

# --- GPU (NVIDIA CUDA) test ---
# Requires: pkg> add CUDA
# and a working NVIDIA driver/CUDA installation
using CUDA
CUDA.allowscalar(false)

println("== CUDA GPU test ==")
# Build on CPU first
I = Index(100, "I")
J = Index(100, "J")
A_cpu = random_itensor(I, J)

# Move to GPU (this triggers ITensor's CUDA extension)
A_gpu = cu(A_cpu)
B_gpu = dag(A_gpu)
C_gpu = A_gpu * B_gpu

@show typeof(A_gpu)       # should show an ITensor with CuArray storage
@show hasinds(C_gpu, prime(I), I)
println("GPU contraction OK ✅")
