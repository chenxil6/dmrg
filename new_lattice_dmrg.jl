using ITensors
using ITensorMPS  
using CSV, DataFrames

using CUDA
CUDA.allowscalar(false)

# ──────────────────────────────────────────────────────────────────────────────
# Interleaved indexing & lattice (matches the "correct" code)
# ──────────────────────────────────────────────────────────────────────────────

# Interleaved indexing: rung j → (2j-1 on leg 1, 2j on leg 2)
site_index(j::Int, leg::Int) = (leg == 1 ? 2*j - 1 : 2*j)

# Build the "correct" lattice: legs (i,i+2), rungs (i,i+1), open boundaries
# function create_lattice(L::Int)
#     N = 2*L
#     lattice = LatticeBond[]               # requires: using ITensorMPS
#     for i in 1:N
#         if i <= N - 2
#             push!(lattice, LatticeBond(i, i + 2))  # leg bond
#         end
#         if i <= N - 1
#             push!(lattice, LatticeBond(i, i + 1))  # rung bond (two types alternate)
#         end
#     end
#     return lattice
# end

function create_lattice(L)

    N = 2*L

    # Define lattice
    lattice = Vector{LatticeBond}()
    # println("N:", N)
    # println(lattice)
    for i in 1:N
        # println(i)
        if i <= N - 2
            lattice = push!(lattice, LatticeBond(i, i + 2))
        end
        if i <= N - 1
            lattice = push!(lattice, LatticeBond(i, i + 1))
        end
        # println(lattice)
    end
    return lattice

end

# Classifiers for bonds in this lattice
is_leg(b::LatticeBond)  = (b.s2 - b.s1) == 2
is_rung(b::LatticeBond) = (b.s2 - b.s1) == 1

# alternating ± sign for rungs (use parity of first site in the pair)
rung_multiplier(p::Int) = iseven(p) ? -1.0 : 1.0

# Coupling for a *pair of sites* (interleaved convention)
# - rung:                    ± J_perp               (no phase)
# - leg (with Peierls phase): -J_parallel * exp(+i * (-mult)*phase), mult from p parity
function coupling_for_pair(p::Int, q::Int; J_perp, Jpar, phase)
    Δ = q - p
    if Δ == 1               # rung-type (two flavors alternate with parity)
        return J_perp
    elseif Δ == 2           # leg bond
        m = rung_multiplier(p)
        return Jpar * exp(1im * (-m) * phase)
    else
        return 0.0 + 0.0im
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian from the "correct" lattice
# ──────────────────────────────────────────────────────────────────────────────
function build_H_from_lattice(sites, lattice; J_perp, Jpar, U, phase)
    os = OpSum()
    for b in lattice
        c = coupling_for_pair(b.s1, b.s2; J_perp=J_perp, Jpar=Jpar, phase=phase)
        # Hermitian hopping with complex c:  c a†_i a_j + c* a†_j a_i
        os += c,       "adag", b.s1, "a", b.s2
        os += conj(c), "adag", b.s2, "a", b.s1
    end
    if U != 0.0
        N = length(sites)
        for i in 1:N
            os += 0.5U, "N", i, "N", i
            os += -0.5U, "N", i
        end
    end
    return MPO(os, sites)
end

function matt_hamiltonian(sites, lattice, psi_0; N, J_perp, J_parallel, U, phase)
    os = OpSum()
    for b in lattice
        if iseven(b.s2 - b.s1)
            multiplier = 1
            if iseven(b.s1)
                multiplier = -1
            end
            os += J_parallel*exp(complex(0,multiplier*phase)), "a", b.s1, "adag", b.s2
            os += J_parallel*exp(complex(0,-multiplier*phase)), "adag", b.s1, "a", b.s2
        else
            os += J_perp, "a", b.s1, "adag", b.s2
            os += J_perp, "adag", b.s1, "a", b.s2
        end
    end

    for i in 1:N
        os += U / 2, "N", i, "N", i
        os -= U / 2, "N", i
    end
    H = MPO(os, sites)
    state  = [isodd(n) ? "0" : "1" for n in 1:N]
    nsweeps = 10
    maxdim = [1600]
    mindim = [1500]
    cutoff = [1E-6]
    noise = [1E-6]
    # psi_ws = random_mps(sites, state)
    energy, psi_0 = dmrg(H, psi_0; nsweeps, mindim, maxdim, cutoff, noise)
    return energy, psi_0
end

# ──────────────────────────────────────────────────────────────────────────────
# Current operators (general bond & pure rung)
# ──────────────────────────────────────────────────────────────────────────────

# General current on a bond (p,q):
#   J_{pq} = -i c a†_p a_q + i c* a†_q a_p
function link_current_os(sites, p::Int, q::Int; J_perp, Jpar, phase)
    c = coupling_for_pair(p, q; J_perp=J_perp, Jpar=Jpar, phase=phase)
    os = OpSum()
    os += -1im*c,        "adag", p, "a", q
    os += +1im*conj(c),  "adag", q, "a", p
    return MPO(os, sites)
end

# Pure rung current with the correct ± sign (no phase on rungs)
function link_rung_current_os(sites, p::Int, q::Int; J_perp)
    os = OpSum()
    os += -1im*J_perp, "adag", p, "a", q
    os += +1im*J_perp, "adag", q, "a", p
    return MPO(os, sites)
end

# ──────────────────────────────────────────────────────────────────────────────
# JJ MPO (works for rungs or legs from the "correct" lattice)
# ──────────────────────────────────────────────────────────────────────────────
function bond_pair_JJ_mpo_local(sites, p,q, r,s; J_perp, Jpar, phase)
    os = OpSum()
    os += -J_perp^2, "adag", p, "a", q, "adag", r, "a", s
    os += +J_perp^2, "adag", p, "a", q, "a",    r, "adag", s
    os += +J_perp^2, "a",    p, "adag", q, "adag", r, "a", s
    os += -J_perp^2, "a",    p, "adag", q, "a",    r, "adag", s
    return MPO(os, sites)
end

# ──────────────────────────────────────────────────────────────────────────────
# One-vector current correlator (self, then successive rungs outward)
# ──────────────────────────────────────────────────────────────────────────────
# EXaCT DUPLICaTE of the correct code’s rung–rung correlator semantics.
# - Raw correlator (no <J1><Jb> subtraction)
# - Start at i=3 (skip (2,3))
# - Flip sign for even i
# - Uses the same JJ 4-term operator with couplings from your helpers
function connected_rung_correlator_first(psi, sites; J_perp, Jpar=0.0, phase=0.0)
    N  = length(sites)
    s11, s12 = 1, 2                 # reference rung (vertical)

    C = Float64[]

    # Self term: <J(1,2) J(1,2)>
    JJ11 = inner(
        psi',
        bond_pair_JJ_mpo_local(sites, s11, s12, s11, s12; J_perp=J_perp, Jpar=Jpar, phase=phase),
        psi
    )
    push!(C, real(JJ11))

    # Now include aLL adjacent pairs to the right starting with (2,3):
    # (2,3) [skew], (3,4) [vertical], ..., (N-1,N)
    for i in 2:(N-1)
        s21, s22 = i, i+1
        JJ = inner(
            psi',
            bond_pair_JJ_mpo_local(sites, s11, s12, s21, s22; J_perp=J_perp, Jpar=Jpar, phase=phase),
            psi
        )
        val = real(JJ)
        # Flip sign for even i to align the orientation of the two rung flavors
        if iseven(i)
            val = -val
        end
        push!(C, val)
    end

    return C
end

function compute_rung_current_correlations(psi, L, J_perp)
    current_correlations = Float64[]

    s11 = 1
    s12 = 2

    for i in 3:2L-1
        operator_sum = OpSum()

        s21 = i
        s22 = i + 1


        # a_i^dagger a_j a_k^dagger a_l
        operator_sum -= J_perp^2, "adag", s11, "a", s12, "adag", s21, "a", s22
        # a_i^dagger a_j a_l^dagger a_k
        operator_sum += J_perp^2, "adag", s11, "a", s12, "a", s21, "adag", s22
        # a_j^dagger a_i a_k^dagger a_l
        operator_sum += J_perp^2, "a", s11, "adag", s12, "adag", s21, "a", s22
        # a_j^dagger a_i a_l^dagger a_k
        operator_sum -= J_perp^2, "a", s11, "adag", s12, "a", s21, "adag", s22


        # Convert the OpSum to an MPO using the same site indices as psi
        operator = MPO(operator_sum, siteinds(psi))
        current_correlation = real(inner(psi', operator, psi))

        if iseven(i)
            current_correlation = -current_correlation
        end
        
        push!(current_correlations, current_correlation)
    end
    return current_correlations
end



# ──────────────────────────────────────────────────────────────────────────────
# Chiral & average rung currents (interleaved convention)
# ──────────────────────────────────────────────────────────────────────────────
function compute_total_chiral_current(psi, lattice, L, J_parallel, phase)
    sites = siteinds(psi)

    os_total = OpSum()
    for j in 1:(L-1)
        i,  ip1  = site_index(j,1),   site_index(j+1,1)   # top leg bond j→j+1
        k,  kp1  = site_index(j,2),   site_index(j+1,2)   # bottom leg bond j→j+1

        # + j_{j,1}^∥  (top leg)
        os_total += -1im*J_parallel*exp(-1im*phase), "adag", i,   "a", ip1
        os_total += +1im*J_parallel*exp(+1im*phase), "adag", ip1, "a", i

        # - j_{j,2}^∥  (bottom leg)
        os_total += +1im*J_parallel*exp(+1im*phase), "adag", k,   "a", kp1
        os_total += -1im*J_parallel*exp(-1im*phase), "adag", kp1, "a", k
    end

    a = MPO(os_total, sites)
    val = inner(psi', a, psi)            # generally complex; small Im part expected
    return val / (2*(L-1))               # average over leg bonds, same normalization
end

function compute_average_rung_current(psi_gpu, sites; J)
    acc = 0.0
    # vertical rungs
    for j in 1:L
        a, b = site_index(j,1), site_index(j,2)
        os = OpSum()
        os += -J, "Adag", a, "A", b
        os += +J, "Adag", b, "A", a
        acc += abs(inner(psi_gpu', MPO(os, sites), psi_gpu))
    end
    # diagonals
    for j in 1:(L-1)
        a2, b = site_index(j+1,1), site_index(j,2)
        os = OpSum()
        os += -J, "Adag", a2, "A", b
        os += +J, "Adag", b,  "A", a2
        acc += abs(inner(psi_gpu', MPO(os, sites), psi_gpu))
    end
    return
end
# ──────────────────────────────────────────────────────────────────────────────
# Sweeps & run_scan (kept coherent with the helpers above)
# ──────────────────────────────────────────────────────────────────────────────
function default_sweeps()
    sw = Sweeps(10)
    setmaxdim!(sw, 800)
    setmindim!(sw, 500)
    setcutoff!(sw, 1e-12)
    setnoise!(sw, 1e-6)
    return sw
end

"""
    run_scan(lattice, L, num_levels, chis; J_perp, J_parallel, U, sweeps=default_sweeps())

Build H from the **correct** lattice each χ, DMRG with warm start, then measure:
- total chiral current on legs
- average rung current magnitude
- (optionally) rung current correlator at χ ≈ π → saved to `rung_corr_chi_pi.csv`

Returns: Vector{NamedTuple} with fields
(:chi, :energy, :Jc_re, :Jc_im, :Jc_abs, :Jr)
"""
function run_scan(lattice::Vector{LatticeBond},
                  L::Int,
                  num_levels::Int,
                  chis;
                  J_perp::Real,
                  J_parallel::Real,
                  U::Real,
                  sweeps::Sweeps = default_sweeps())

    N = 2*L
    sites = siteinds("Boson", N; dim=num_levels, conserve_qns=true)

    # half-filling product state matched to interleaved indexing: 1→"0", 2→"1", ...
    state  = [isodd(n) ? "0" : "1" for n in 1:N]
    

    rows = NamedTuple[]
    psi_0 = random_mps(sites, state)
    for χ in chis
        
        # 1) Build H from the correct lattice (interleaved sites)
        # H = build_H_from_lattice(sites, lattice; J_perp=J_perp, Jpar=J_parallel, U=U, phase=χ)
        energy, psi_ws = matt_hamiltonian(sites, lattice, psi_0; N, J_perp=J_perp, J_parallel = J_parallel, U=U, phase=χ)

        # 2) DMRG with warm start
        # energy, psi_ws = dmrg(H, psi_ws, sweeps)

        # 3) Measurements
        Jc = compute_total_chiral_current(psi_ws, lattice, L, J_parallel, χ)
        Jr = compute_average_rung_current(psi_ws, sites; J = J_perp)

        # 4) Optional: rung current correlator at χ ≈ π
        if isapprox(χ, Base.MathConstants.pi; atol=1e-8)
            # C = connected_rung_correlator_first(psi_ws, sites; J_perp=J_perp, Jpar=J_parallel, phase=χ)
            C = compute_rung_current_correlations(psi_ws, L, J_perp)/J_perp^2
            CSV.write("rung_corr_chi_pi.csv",
                DataFrame(
                    rung = 1:length(C),
                    Cre  = C,
                    Cim  = zeros(length(C)),
                    Cabs = abs.(C)
                )
            )
        end

        push!(rows, (
            chi    = χ,
            energy = real(energy),
            Jc_re  = real(Jc),
            Jc_im  = imag(Jc),
            Jc_abs = abs(Jc),
            Jr     = Jr,
        ))
    end

    return rows
end

# ──────────────────────────────────────────────────────────────────────────────
# Example main (keep your own parameter values as before)
# ──────────────────────────────────────────────────────────────────────────────
L = 5
num_levels = 5
J_perp = -1
J_parallel = -0.5
J_ratio = J_parallel/J_perp
U = 25
N=2*L

lattice = create_lattice(L)
chis    = range(0, stop=Base.MathConstants.pi, length=11)
rows = run_scan(lattice, L, num_levels, chis; J_perp=J_perp, J_parallel=J_parallel, U=U)

# sites = siteinds("Boson", N; dim=num_levels, conserve_qns=true)
# state  = [isodd(n) ? "0" : "1" for n in 1:N]
# psi_ws = random_mps(sites, state)
# H = matt_hamiltonian(sites, lattice; N, J_perp=J_perp, J_parallel = J_parallel, U=U, phase=Base.MathConstants.pi)
# sweeps = default_sweeps()
# E, psi0 = dmrg(H, psi_ws, sweeps)
# C = compute_rung_current_correlations(psi0, L, J_perp)
# CSV.write("rung_corr_chi_pi.csv",
#     DataFrame(
#         rung = 1:length(C),
#         Cre  = C,
#         Cim  = zeros(length(C)),
#         Cabs = abs.(C)
#     )
# )

df   = DataFrame(rows)
fname = "ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv"
CSV.write(fname, df)
println("Wrote $fname")
