using ITensors
using ITensorMPS  
using CSV, DataFrames

using CUDA
CUDA.allowscalar(false)

# --- params ---
L      = 4
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

nsweeps = 10
sweeps  = Sweeps(nsweeps)
setmaxdim!(sweeps, 500)
# setmindim!(sweeps,  200)
setcutoff!(sweeps, 1e-10)

chis = range(0, stop=π, length=10)
rows = Vector{NamedTuple}()

function link_current_os(p, q, sites; J, phi=0.0)
    os = OpSum()
    os += -1im*J*exp( +1im*phi), "Adag", p, "A", q
    os += +1im*J*exp( -1im*phi), "A", p, "Adag", q
    return cu(MPO(os, sites))
end

# MPO for the product J_1 * J_j written explicitly as 4 terms
    # --- small helper: JJ MPO for arbitrary bonds (p,q) and (r,s)
function bond_pair_JJ_mpo_local(p,q,r,s; phi1=0,phi2=0)
    os = OpSum()
    # (-) e^{ i(φ+φ)} a†_p a_q a†_r a_s
    os += -J^2 * exp(1im*(phi1+phi2)),  "Adag", p, "A", q, "Adag", r, "A", s
    # (+) e^{ i(φ-φ)} a†_p a_q a†_s a_r
    os += +J^2 * exp(1im*(phi1-phi2)),  "Adag", p, "A", q, "A", r, "Adag", s
    # (+) e^{ i(-φ+φ)} a†_q a_p a†_r a_s
    os += +J^2 * exp(1im*(-phi1+phi2)), "A", p, "Adag", q, "Adag", r, "A", s
    # (-) e^{-i(φ+φ)} a†_q a_p a†_s a_r
    os += -J^2 * exp(-1im*(phi1+phi2)), "A", p, "Adag", q, "A", r, "Adag", s
    return cu(MPO(os, sites))
end

# Connected correlator { <J1 J_b> - <J1><J_b> } including self and diagonals.
# Order: [ self, rung(2), diag(1↔2), rung(3), diag(2↔3), …, rung(L), diag(L-1↔L) ]
function connected_rung_correlator_first(psi, sites; J, phi1=0.0, phi2=0.0)
    Lr = length(sites) ÷ 2
    # --- J on first rung (1,1)↔(1,2)
    p1, q1 = site_index(1,1), site_index(1,2)
    A1 = link_current_os(p1, q1, sites; J=J, phi=0)
    J1 = inner(psi', A1, psi)

    C = Float64[]

    # 1) self: C11 = <J1^2> - <J1>^2
    JJ11 = inner(psi', bond_pair_JJ_mpo_local(p1,q1,p1,q1;phi1,phi2), psi)
    push!(C, real(JJ11 - J1*J1))

    # 2) alternate rung(n) and diag(j) moving outwards
    for n in 1:Lr
        if n == 1
            continue
        else
            pd, qd = site_index(n,1), site_index(n-1,2)
            Ad = link_current_os(pd, qd, sites; J=J, phi=0)
            Jd = inner(psi', Ad, psi)
            JJd = inner(psi', bond_pair_JJ_mpo_local(p1,q1,pd,qd;phi1,phi2), psi)
            push!(C, real(JJd - J1*Jd))
        end
        # rung n: (n,1)↔(n,2)
        pr, qr = site_index(n,1), site_index(n,2)
        Ar = link_current_os(pr, qr, sites; J=J, phi=phi2)
        Jr = inner(psi', Ar, psi)
        JJr = inner(psi', bond_pair_JJ_mpo_local(p1,q1,pr,qr;phi1,phi2), psi)
        push!(C, real(JJr - J1*Jr))

    end
    return C
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
            C  = connected_rung_correlator_first(psi_ws, sites; J=J, phi1 = 0,phi2=0)
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
