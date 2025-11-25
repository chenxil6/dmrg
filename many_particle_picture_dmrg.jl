using ITensors
using ITensorMPS  
using CSV, DataFrames

using CUDA
CUDA.allowscalar(false)

# --- params ---
L      = 25
rho      = 0.5                      # half filling like the paper
N      = Int(round(rho*2L))         # total bosons

Nmax   = 3                        # paper used ≥4–5
J, Jpar, U = 1.0, 0.5, 25        # match paper examples
J_ratio = Jpar/J;
sites  = siteinds("Boson", 2*L; dim=Nmax+1, conserve_qns=true)

nsweeps = 10
sweeps  = Sweeps(nsweeps)

setmaxdim!(sweeps, 1000)
# setmindim!(sweeps, 500)
setcutoff!(sweeps, 1e-12)
setnoise!(sweeps,  1e-6) # <- crucial for growth
# nsweeps = 12
# sweeps  = Sweeps(nsweeps)
# setmaxdim!(sweeps, 100, 200, 400, 800, 1200, 1600, 2000, 2400, 2400, 2400, 2400, 2400)
# setcutoff!(sweeps, 1e-8)
# setnoise!(sweeps, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

chis = range(0, stop=Base.MathConstants.pi, length=21)
rows = Vector{NamedTuple}()

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

    A_gpu = MPO(os_total, sites)
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

psi0 = random_mps(sites, half_filled_state())

function link_current_os(sites, p, q; J)
    os = OpSum()
    os += -1im*J, "Adag", p, "A", q
    os += +1im*J, "A", p, "Adag", q
    return MPO(os, sites)
end
function rung_multiplier(p::Int, q::Int, _sites)
    m = 1
    if iseven(q)
        m = -1
    end        
end
# MPO for the product J_1 * J_j written explicitly as 4 terms
    # --- small helper: JJ MPO for arbitrary bonds (p,q) and (r,s)
function bond_pair_JJ_mpo_local(sites, p,q,r,s; J)
    # Build rung couplings with alternating sign

    os = OpSum()
    # <J_pq J_rs> expanded into 4 normal-ordered terms using c1,c2
    os += -J^2,             "Adag", p, "A", q, "Adag", r, "A", s
    os += J^2,       "Adag", p, "A", q, "A", r, "Adag", s
    os += J^2,       "A",    p, "Adag", q, "Adag", r, "A", s
    os += -J^2, "A",    p, "Adag", q, "A", r, "Adag", s
    return MPO(os, sites)
end


function connected_rung_correlator_first(psi, sites; J)
    Lr = length(sites) ÷ 2

    # Reference rung: (1,1) ↔ (1,2)
    p1, q1 = site_index(1,1), site_index(1,2)
    A1     = link_current_os(sites, p1, q1; J = J)
    J1     = inner(psi', A1, psi)

    C = Float64[]

    # self: <J1^2> - <J1>^2
    JJ11 = inner(psi', bond_pair_JJ_mpo_local(sites, p1, q1, p1, q1; J=J), psi)
    push!(C, real(JJ11 - J1*J1))/(J^2)

    # For j = 1..Lr-1, append "skew" then "vertical" rung at distance j
    for j in 1:(Lr-1)
        # skew: (j+1,1) ↔ (j,2)
        pe, qe = site_index(j+1,1), site_index(j,2)
        Ae     = link_current_os(sites, pe, qe; J = J)
        Je     = inner(psi', Ae, psi)
        JJe    = inner(psi', bond_pair_JJ_mpo_local(sites, p1, q1, pe, qe; J=J), psi)
        push!(C, real(JJe - J1*Je))/(J^2)
        
        # vertical: (j+1,1) ↔ (j+1,2)
        po, qo = site_index(j+1,1), site_index(j+1,2)
        Ao     = link_current_os(sites, po, qo; J = J)
        Jo     = inner(psi', Ao, psi)
        JJo    = inner(psi', bond_pair_JJ_mpo_local(sites, p1, q1, po, qo; J=J), psi)
        push!(C, real(JJo - J1*Jo))/(J^2)
    end

    return C
end

function run_scan(sites, chis; J, Jpar, U, sweeps, psi0)
    
    rows = NamedTuple[]

    for χ in chis
        # psi_ws = random_mps(sites, half_filled_state())
        psi_ws = psi0   # GPU warm-start
        # Build H on CPU, then move to GPU
        H_cpu = build_H(sites; χ=χ, J=J, Jpar=Jpar, U=U)
        H_gpu = H_cpu

        # DMRG on GPU
        E, psi_gs = dmrg(H_gpu, psi_ws, sweeps)

        psi_a = psi_gs
        psi_b = psi_gs
        psi_c = psi_gs
        # Measurements (GPU MPOs)
        Jc = chiral_current_gpu(psi_a, sites; χ=χ, Jpar=Jpar)
        Jr = avg_rung_current_gpu(psi_b, sites; J=J)

        # Optional: correlator at χ = π
        if isapprox(χ, Base.MathConstants.pi; atol=1e-12)
            C  = connected_rung_correlator_first(psi_c, sites; J)
            CSV.write("rung_corr_chi_pi.csv",
                DataFrame(rung=1:length(C),
                Cre=real.(C),
                Cim=imag.(C),
                Cabs=abs.(C))
            )
        end

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
