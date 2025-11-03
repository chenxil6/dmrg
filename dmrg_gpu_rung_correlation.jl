using ITensors
using ITensorMPS  
using CSV, DataFrames

using CUDA
CUDA.allowscalar(false)

# --- params ---
L      = 5
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
setmaxdim!(sweeps, 800)
# setmindim!(sweeps,  200)
setcutoff!(sweeps, 1e-6)

chis = range(0, stop=π, length=20)
rows = Vector{NamedTuple}()

"""
    bond_coupling(s1, s2; J_perp, J_parallel, phase)

Implements your "correct code" rule:
- multiplier = (-1) if s1 is even, else (+1)
- If (s2 - s1) is even => treat as a leg bond:  coupling = -J_parallel * exp(-i*multiplier*phase)
- Else                 => treat as a rung/diagonal: coupling = multiplier * J_perp

NOTE: With site_index(j,m)=(m-1)L + j and L=5 (odd):
  - Rungs have (s2-s1)=L (odd)  -> "rung/diag" branch
  - Legs  have (s2-s1)=±1 (even)-> "leg" branch
If you switch to even L, this parity test will flip for rungs; adjust as needed.
"""
function bond_coupling(s1::Int, s2::Int; J_perp, J_parallel, phase)
    mult = iseven(s1) ? -1 : 1
    if iseven(s2 - s1)
        # leg bond
        return -J_parallel * exp(-1im * mult * phase)
    else
        # rung/diagonal bond
        return mult * J_perp
    end
end

"""
    single_bond_current_mpo(s1, s2, sites; coupling, to_cuda=false)

Defines J_(s1,s2) so that J_b1 * J_b2 expands to the 4 terms with coefficients:
  - c1*c2, + c1*conj(c2), + conj(c1)*c2, - conj(c1)*conj(c2)
which is exactly your "correct code" pattern.
"""
function single_bond_current_mpo(s1::Int, s2::Int, sites; coupling, to_cuda::Bool=false)
    os = OpSum()
    # J_b = (-c) a†_s1 a_s2  +  (conj(c)) a†_s2 a_s1
    os += -coupling,       "Adag", s1, "A", s2
    os +=  conj(coupling), "Adag", s2, "A", s1
    mpo = MPO(os, sites)
    return to_cuda ? cu(mpo) : mpo
end

"""
    bond_pair_JJ_mpo(b1, b2, sites; c1, c2, to_cuda=false)

Explicit 4-term MPO for J_(b1) * J_(b2), with the same coefficients and conjugations
as your "correct code". Bonds are NamedTuples like (; s1=Int, s2=Int).
"""
function bond_pair_JJ_mpo(b1, b2, sites; c1, c2, to_cuda::Bool=false)
    i, j = b1.s1, b1.s2
    k, l = b2.s1, b2.s2

    os = OpSum()
    # (-) c1*c2            a†_i a_j a†_k a_l
    os += -(c1*c2),                "Adag", i, "A", j, "Adag", k, "A", l
    # (+) c1*conj(c2)      a†_i a_j a†_l a_k
    os += +(c1*conj(c2)),          "Adag", i, "A", j, "Adag", l, "A", k
    # (+) conj(c1)*c2      a†_j a_i a†_k a_l
    os += +(conj(c1)*c2),          "Adag", j, "A", i, "Adag", k, "A", l
    # (-) conj(c1)*conj(c2) a†_j a_i a†_l a_k
    os += -(conj(c1)*conj(c2)),    "Adag", j, "A", i, "Adag", l, "A", k

    mpo = MPO(os, sites)
    return to_cuda ? cu(mpo) : mpo
end

"""
    build_lattice_bonds()

Collect ALL bonds used by your Hamiltonian (legs, rungs, diagonals) as
NamedTuples (; s1, s2) with s1 < s2 to keep them canonical.
"""
function build_lattice_bonds()
    bonds = NamedTuple{(:s1,:s2),Tuple{Int,Int}}[]

    # Legs (with direction j -> j+1 on each leg)
    for j in 1:L-1
        i1,i2 = site_index(j,1), site_index(j+1,1)
        k1,k2 = site_index(j,2), site_index(j+1,2)
        push!(bonds, (s1=min(i1,i2), s2=max(i1,i2)))
        push!(bonds, (s1=min(k1,k2), s2=max(k1,k2)))
    end

    # Rungs
    for j in 1:L
        a,b = site_index(j,1), site_index(j,2)
        push!(bonds, (s1=min(a,b), s2=max(a,b)))
    end

    # Diagonals
    for j in 1:L-1
        a,b = site_index(j+1,1), site_index(j,2)
        push!(bonds, (s1=min(a,b), s2=max(a,b)))
    end

    # Make unique
    unique!(bonds)
    return bonds
end

"""
    connected_current_correlators(psi, lattice, sites;
        J_perp, J_parallel, phase, q1::Int=0, q2::Int=0, to_cuda::Bool=true)

Compute connected ⟨J_b1 J_b2⟩ - ⟨J_b1⟩⟨J_b2⟩ over all pairs in `lattice`
with your correct-code rules:
  - Optional filter on the *first* bond: if q1,q2≠0 require (b1.s1==q1 && b1.s2==q2)
  - Skip any pair that shares a site
  - Only keep ordered pairs with b2.s1 ≥ b1.s1 (prevents double counting)
Returns a vector of NamedTuples (b1_s1,b1_s2,b2_s1,b2_s2,value).
"""
function connected_current_correlators(psi, lattice, sites;
        J_perp, J_parallel, phase, q1::Int=0, q2::Int=0, to_cuda::Bool=true)

    out = NamedTuple{(:b1_s1,:b1_s2,:b2_s1,:b2_s2,:value),
                     Tuple{Int,Int,Int,Int,Float64}}[]

    for b1 in lattice
        # Optional (q1,q2) filter on the first bond
        if (q1 != 0 && q1 != b1.s1) || (q2 != 0 && q2 != b1.s2)
            continue
        end

        # Coupling and <J_b1>
        c1    = bond_coupling(b1.s1, b1.s2; J_perp=J, J_parallel=Jpar, phase=phase)
        Jb1op = single_bond_current_mpo(b1.s1, b1.s2, sites; coupling=c1, to_cuda=to_cuda)
        Jb1   = inner(psi', Jb1op, psi)

        for b2 in lattice
            # Skip if any site is shared
            if b1.s1 == b2.s1 || b1.s1 == b2.s2 || b1.s2 == b2.s1 || b1.s2 == b2.s2
                continue
            end
            # Ordering to avoid duplicates
            if b2.s1 < b1.s1
                continue
            end

            c2    = bond_coupling(b2.s1, b2.s2; J_perp=J, J_parallel=Jpar, phase=phase)
            Jb2op = single_bond_current_mpo(b2.s1, b2.s2, sites; coupling=c2, to_cuda=to_cuda)
            Jb2   = inner(psi', Jb2op, psi)

            JJop  = bond_pair_JJ_mpo(b1, b2, sites; c1=c1, c2=c2, to_cuda=to_cuda)
            JJ    = inner(psi', JJop, psi)

            Cconn = real(JJ - Jb1*Jb2)
            push!(out, (b1.s1, b1.s2, b2.s1, b2.s2, Cconn))
        end
    end
    return out
end



function half_filled_mps(sites, N)
    state = fill("0", length(sites))
    for i in 1:N
        state[i] = "1"
    end
    return MPS(sites, state)
end

function run_scan(sites, chis; J, Jpar, U, sweeps, psi0)
    psi_ws = map(cu, psi0)   # GPU warm-start
    rows = NamedTuple[]

    for χ in chis
        # Build H on CPU, then move to GPU
        H_cpu = build_H(sites; χ=χ, J=J, Jpar=Jpar, U=U)
        H_gpu = cu(H_cpu)

        # DMRG on GPU
        E, psi_ws = dmrg(H_gpu, psi_ws, sweeps)

        # Measurements (GPU MPOs)
        Jc = chiral_current_gpu(psi_ws, sites; χ=χ, Jpar=Jpar)
        Jr = avg_rung_current_gpu(psi_ws, sites; J=J)

        # Optional: correlator at χ = π
        if isapprox(χ, π; atol=1e-12)
            lattice = build_lattice_bonds()
            CC = connected_current_correlators(
                    psi_ws, lattice, sites;
                    J_perp=J, J_parallel=Jpar, phase=χ, to_cuda=true)

            dfC = DataFrame(
                b1_s1 = [c.b1_s1 for c in CC],
                b1_s2 = [c.b1_s2 for c in CC],
                b2_s1 = [c.b2_s1 for c in CC],
                b2_s2 = [c.b2_s2 for c in CC],
                Cconn = [c.value for c in CC],
                )
            CSV.write("bond_corr_chi_pi.csv", dfC)

        push!(rows, (
            chi    = Float64(χ),
            energy = Float64(real(E)),
            Jc_re  = Float64(real(Jc)),
            Jc_im  = Float64(imag(Jc)),
            Jc_abs = Float64(abs(Jc)),
            Jr     = Float64(Jr),
        ))
    end
    return rows
end

rows = run_scan(sites, chis; J=J, Jpar=Jpar, U=U, sweeps=sweeps, psi0=psi0)
df   = DataFrame(rows)  # ← multiple rows as intended
fname = "ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv"
CSV.write(fname, df)
println("Wrote $fname")
