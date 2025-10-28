using ITensors
using ITensorMPS  
using CSV, DataFrames

using CUDA
CUDA.allowscalar(false)

# --- params ---
L      = 15
rho      = 0.5                      # half filling like the paper
N      = Int(round(rho*2L))         # total bosons

Nmax   = 4                        # paper used ≥4–5
J, Jpar, U = 1.0, 0.5, 2.5        # match paper examples
J_ratio = Jpar/J;
sites  = siteinds("Boson", 2L; dim=Nmax+1, conserve_number=true)

site_index(j,m) = (m-1)*L + j  # map (rung j, leg m∈{1,2}) -> 1..2L

function build_H(sites; χ, J=1.0, Jpar=0.5, U=0.0)
    os = OpSum()
    # legs with Peierls phases
    for j in 1:L-1
        i1,i2 = site_index(j,1), site_index(j+1,1)
        k1,k2 = site_index(j,2), site_index(j+1,2)
        os += -Jpar*exp(-im*χ), "Adag", i1, "A", i2
        os += -Jpar*exp(+im*χ), "Adag", i2, "A", i1
        os += -Jpar*exp(+im*χ), "Adag", k1, "A", k2
        os += -Jpar*exp(-im*χ), "Adag", k2, "A", k1
    end
    # rungs
    for j in 1:L
        a,b = site_index(j,1), site_index(j,2)
        os += -J, "Adag", a, "A", b
        os += -J, "Adag", b, "A", a
    end
    # diagonals
    for j in 1:L-1
        a,b = site_index(j+1,1), site_index(j,2)
        os += -J, "Adag", a, "A", b
        os += -J, "Adag", b, "A", a
    end
    # onsite interaction  (Boson site type supports "N(N-1)")
    if U != 0.0
        for i in 1:2L
            os += 0.5U, "N", i, "N", i   # (U/2) * N^2
            os += -0.5U, "N", i          # -(U/2) * N
        end
    end
    return MPO(os, sites)
end

function avg_rung_current_gpu(psi_gpu, sites; J)
    acc = 0.0
    # vertical rungs
    for j in 1:L
        a, b = site_index(j,1), site_index(j,2)
        os = OpSum()
        os += -1im*J, "Adag", a, "A", b
        os += +1im*J, "Adag", b, "A", a
        acc += abs(inner(psi_gpu', cu(MPO(os, sites)), psi_gpu))
    end
    # diagonals
    for j in 1:(L-1)
        a2, b = site_index(j+1,1), site_index(j,2)
        os = OpSum()
        os += -1im*J, "Adag", a2, "A", b
        os += +1im*J, "Adag", b,  "A", a2
        acc += abs(inner(psi_gpu', cu(MPO(os, sites)), psi_gpu))
    end
    return acc/(2L - 1)
end


function chiral_current_gpu(psi_gpu, sites; χ, Jpar)
    os_total = OpSum()
    for j in 1:L-1
        i, ip1 = site_index(j,1), site_index(j+1,1)
        k, kp1 = site_index(j,2), site_index(j+1,2)

        # + j_{j,1}^||
        os_total += -1im*Jpar*exp(-1im*χ), "Adag", i,   "A", ip1
        os_total += +1im*Jpar*exp(+1im*χ), "Adag", ip1, "A", i
        # - j_{j,2}^||
        os_total += +1im*Jpar*exp(+1im*χ), "Adag", k,   "A", kp1
        os_total += -1im*Jpar*exp(-1im*χ), "Adag", kp1, "A", k
    end

    A_gpu = cu(MPO(os_total, sites))
    # Either:
    # val = inner(psi_gpu', A_gpu, psi_gpu)
    val = inner(psi_gpu', A_gpu, psi_gpu)
    val = ComplexF64.(val)
    return val / (2*(L-1))
end

function half_filled_state()
    state = fill("0", 2L)
    for i in 1:N
        state[i] = "1"            # ensure total N; respects QN conservation
    end
    return state
end

psi0 = MPS(sites, half_filled_state())

nsweeps = 1
sweeps  = Sweeps(nsweeps)
setmaxdim!(sweeps, 500)
# setmindim!(sweeps,  1000)
setcutoff!(sweeps, 1e-6)

chis = range(0, stop=π, length=20)
rows = Vector{NamedTuple}()

function half_filled_mps(sites, N)
    state = fill("0", length(sites))
    for i in 1:N
        state[i] = "1"
    end
    return MPS(sites, state)
end

function run_scan(sites, chis; J, Jpar, U, sweeps, psi0)
    # Start with a GPU copy of the initial MPS
    psi_ws = map(cu, psi0)              # <-- GPU MPS (warm-start)
    rows = NamedTuple[]

    for χ in chis
        # Build H on CPU, then move to GPU
        H_cpu = build_H(sites; χ, J, Jpar, U)
        H_gpu = cu(H_cpu)                # <-- GPU MPO

        # DMRG entirely on GPU
        E, psi_ws = dmrg(H_gpu, psi_ws, sweeps)

        # Measure with GPU MPOs too (avoid CPU/GPU mixing):
        # Build the small MPOs on CPU, then cu(...) them for inner(...)
        Jc = chiral_current_gpu(psi_ws, sites; χ, Jpar=Jpar)
        Jr = avg_rung_current_gpu(psi_ws, sites; J=J)

        push!(rows, (chi=χ, energy=E, Jc=Jc, Jr=Jr))
    end
    return rows
end

rows = run_scan(sites, chis; J=J, Jpar=Jpar, U=U, sweeps=sweeps, psi0=psi0)

df = DataFrame(rows)
fname = "ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv"
CSV.write(fname, df)
println("Wrote ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv")