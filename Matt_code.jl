using ITensors, ITensorMPS
ENV["GKSwstype"] = "100"   # offscreen GR; required in VS Code/WSL/remote



using Plots
gr()                       # VS Code-friendly backend
default(show = true)       # auto-display every plot / plot! / scatter / scatter!
using LsqFit
using FFTW
using JLD2
using Dates
using CSV, DataFrames

function compute_ground_state(lattice, L, num_levels, J_perp, J_parallel, U, phase)
    # Number of qubits on each leg
    N = 2 * L

    # DMRG parameters
    nsweeps = 5
    maxdim = [100, 200, 400, 800, 1600]
    mindim = [50,100,200,300,400]
    cutoff = [1E-6]
    noise = [1E-6, 1E-7, 1E-8, 0.0]

    # Define sites
    sites = siteinds("Boson", 2 * L; dim=num_levels, conserve_qns=true)

    # Define Hamiltonian
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

    # Half filling
    state = [isodd(n) ? "0" : "1" for n in 1:N]

    # Initialize wavefunction to a random MPS
    # of bond-dimension 10 with same quantum
    # numbers as `state`
    psi0 = random_mps(sites, state)

    # Perform DMRG
    energy, psi = dmrg(H, psi0; nsweeps, mindim, maxdim, cutoff, noise)

    # Return ground state energy and wavefunction
    return energy, psi
end


function create_lattice(L)

    N = 2*L

    # Define lattice
    lattice = Vector{LatticeBond}()
    println("N:", N)
    println(lattice)
    for i in 1:N
        println(i)
        if i <= N - 2
            lattice = push!(lattice, LatticeBond(i, i + 2))
        end
        if i <= N - 1
            lattice = push!(lattice, LatticeBond(i, i + 1))
        end
        println(lattice)
    end
    return lattice

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
        current_correlation = real(inner(psi, operator, psi))

        if iseven(i)
            current_correlation = -current_correlation
        end
        
        push!(current_correlations, current_correlation)
    end

    return current_correlations
end

function compute_current_correlations(psi, lattice, L, J_perp, J_parallel, phase, q1, q2)
    current_correlations = Float64[]
    for b1 in lattice

        # skip if q1 and q2 are provided and not equal to b1.s1 or b1.s2
        if (q1 != 0 && q1 != b1.s1) || (q2 != 0 && q2 != b1.s2)
            # println("Skipping bond $(b1.s1)-$(b1.s2) due to q1 and q2 conditions")
            continue
        end

        multiplier_1 = 1
        if iseven(b1.s1)
            multiplier_1 = -1
        end
        if iseven(b1.s2 - b1.s1)
            coupling_1 = -J_parallel*exp(complex(0,-multiplier_1*phase))
        else
            coupling_1 = multiplier_1 * J_perp
        end

        for b2 in lattice
            operator_sum = OpSum()

            # skip of any two sites are the same
            if b1.s1 == b2.s1 || b1.s1 == b2.s2 || b1.s2 == b2.s1 || b1.s2 == b2.s2
                continue
            end

            # skip if starting index of second bond is less than starting index of first bond
            if b2.s1 < b1.s1
                continue
            end
            
            multiplier_2 = -1
            if iseven(b2.s1)
                multiplier_2 = 1
            end
            if iseven(b2.s2 - b2.s1)
                coupling_2 = -J_parallel*exp(complex(0,-multiplier_2*phase))
            else
                coupling_2 = multiplier_2 * J_perp
            end
            if b2.s2 - b2.s1 > 1
                continue
            end

            # a_i^dagger a_j a_k^dagger a_l
            operator_sum -= coupling_1*coupling_2, "adag", b1.s1, "a", b1.s2, "adag", b2.s1, "a", b2.s2
            # a_i^dagger a_j a_l^dagger a_k
            operator_sum += coupling_1*conj(coupling_2), "adag", b1.s1, "a", b1.s2, "a", b2.s1, "adag", b2.s2
            # a_j^dagger a_i a_k^dagger a_l
            operator_sum += conj(coupling_1)*coupling_2, "a", b1.s1, "adag", b1.s2, "adag", b2.s1, "a", b2.s2
            # a_j^dagger a_i a_l^dagger a_k
            operator_sum -= conj(coupling_1)*conj(coupling_2), "a", b1.s1, "adag", b1.s2, "a", b2.s1, "adag", b2.s2
            
            operator = MPO(operator_sum, siteinds(psi))
            current_correlation = real(inner(psi, operator, psi))/J_perp^2
            push!(current_correlations, current_correlation)
            println("Current correlation for bond $(b1.s1)-$(b1.s2) to bond $(b2.s1)-$(b2.s2): ", current_correlation)
        end
    end
    CSV.write("rung_corr_chi_pi_at_J_parallel_=$(J_parallel).csv",
    DataFrame(
        rung = 1:length(current_correlations),
        Cre  = current_correlations,
        Cim  = zeros(length(current_correlations)),
        Cabs = abs.(current_correlations)
    ))
end

L = 16
num_levels = 4
J_parallel = -0.1
J_perp = -1
phase = 1 * Base.MathConstants.pi
U = 25
J_para_list = [-0.1, -0.5, -1, -1.5, -2]
lattice = create_lattice(L)
for J_parallel in J_para_list
    energy, psi = compute_ground_state(lattice, L, num_levels, J_perp, J_parallel, U, phase)
    println("J_parallel = $(J_parallel)")

    compute_current_correlations(psi, lattice, L, J_perp, J_parallel, phase, 1, 2)
end


