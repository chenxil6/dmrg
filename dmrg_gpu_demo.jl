# dmrg_gpu_demo.jl

using ITensors, ITensorMPS

# --- Model setup (CPU) ---
N = 50                          # chain length (increase for heavier runs)
s = siteinds("S=1/2", N)

# Heisenberg XXZ: H = Σ Sz_i Sz_{i+1} + 0.5(S+_i S-_i+1 + S-_i S+_i+1)
ampo = AutoMPO()
for n in 1:N-1
  add!(ampo, "Sz", n, "Sz", n+1)
  add!(ampo, 0.5, "S+", n, "S-", n+1)
  add!(ampo, 0.5, "S-", n, "S+", n+1)
end
H = MPO(ampo, s)

# initial MPS with moderate bond dimension
psi0 = randomMPS(s, 10)

# --- Sweeps configuration ---
sweeps = Sweeps(6)                     # number of DMRG sweeps
setmaxdim!(sweeps, 50, 100, 200, 400, 800, 1000)
setcutoff!(sweeps, 1e-9)
setnoise!(sweeps, 1e-8, 1e-8, 1e-9, 0.0, 0.0, 0.0)   # light noise early can help convergence

println("=== CPU DMRG (baseline) ===")
E_cpu, psi_cpu = dmrg(H, psi0, sweeps)
println("CPU ground-state energy: ", E_cpu)

# ----------------------------
# --- Choose ONE GPU path ----
# ----------------------------

# ===== NVIDIA CUDA =====
# ] add CUDA  (once, in Pkg)
using CUDA
CUDA.allowscalar(false)

println("\n=== CUDA GPU DMRG ===")
H_gpu   = map(cu, H)           # move every tensor in MPO to GPU
psi_gpu = map(cu, psi0)        # move MPS to GPU

E_gpu, psi_gpu = dmrg(H_gpu, psi_gpu, sweeps)
println("CUDA ground-state energy: ", E_gpu)

