#!/usr/bin/env python3

import os
from pathlib import Path

from flax import nnx

import jax
import jax.numpy as jnp
import numpy as onp
from jax import jit, grad, random, lax, config as jax_config_mod

jax_config_mod.update("jax_enable_x64", True)

from jax_md import simulate, quantity, dataclasses, units, space, partition

from mace.calculators import foundations_models
from mace.tools.scripts_utils import extract_config_mace_model
from jax_md._nn.mace.mace_jax_from_torch import convert_model
from jax_md._nn.mace.featurizer import mace_featurizer
from mace_jax.tools.device import configure_torch_runtime, get_torch_device

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


DTYPE = jnp.float64


def load_foundation_model(source="mp", variant=None, device="cpu"):
    loader_kwargs = {"device": device}
    src = source.lower()

    if src in {"mp", "off", "omol"}:
        loader = getattr(foundations_models, f"mace_{src}")
        if variant is not None:
            loader_kwargs["model"] = variant
    elif src == "anicc":
        loader = foundations_models.mace_anicc
        if variant is not None:
            loader_kwargs["model_path"] = variant
    else:
        raise ValueError(f"Unknown foundation source: {source!r}")

    model = loader(return_raw_model=True, **loader_kwargs)
    return model.float().eval()


def read_lammps_dump_timestep(filename, target_timestep=0):
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        timestep = int(lines[i + 1].strip())
        n_atoms = int(lines[i + 3].strip())

        atoms_header = lines[i + 8].strip()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"Unexpected dump format near line {i + 9}: {atoms_header}")

        columns = atoms_header.split()[2:]
        data_start = i + 9
        data_end = data_start + n_atoms

        if timestep == target_timestep:
            rows = [line.split() for line in lines[data_start:data_end]]
            data = onp.array(rows, dtype=float)
            col = {name: idx for idx, name in enumerate(columns)}

            ids = data[:, col["id"]].astype(int)
            types = data[:, col["type"]].astype(int)
            pos = onp.stack([data[:, col["x"]], data[:, col["y"]], data[:, col["z"]]], axis=1)
            vel = onp.stack([data[:, col["vx"]], data[:, col["vy"]], data[:, col["vz"]]], axis=1)
            force = onp.stack([data[:, col["fx"]], data[:, col["fy"]], data[:, col["fz"]]], axis=1)

            order = onp.argsort(ids)
            return {
                "ids": ids[order],
                "types": types[order],
                "pos": pos[order],
                "vel": vel[order],
                "force": force[order],
            }

        i = data_end

    raise ValueError(f"Timestep {target_timestep} not found in {filename}")


def main():
    output_dir = Path("output_nve_si_mace")
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_file = "step_1.traj"
    dump_file = "dump.nve"
    thermo_file = "thermo_mace.dat"

    smoke_test = os.environ.get("READTHEDOCS", "False").lower() in ("1", "true", "yes")
    nsteps_sim = 1500 if smoke_test else 50000
    write_every = 100
    verbose = True

    foundation_source = "mp"
    foundation_variant = "small-0b2"
    k_neighbors = 96
    dr_threshold = 0.5
    capacity_multiplier = 6.0

    box_np = onp.array([21.724, 21.724, 21.724], dtype=onp.float32)
    box = jnp.array(box_np, dtype=DTYPE)
    box_matrix = jnp.diag(box)

    timestep_ps = 1e-3

    data_lammps = pd.read_csv(
        thermo_file,
        comment="#",
        delimiter=r"\s+",
        header=None,
    )
    data_lammps.columns = ["Step", "T", "P", "E", "K", "H"]

    t_l = data_lammps["Step"].to_numpy() * timestep_ps
    T_lammps = data_lammps["T"].to_numpy()
    P_lammps = data_lammps["P"].to_numpy()
    E_lammps = data_lammps["E"].to_numpy()
    H_lammps = data_lammps["H"].to_numpy()

    lammps_step_0 = onp.loadtxt(traj_file, dtype=onp.float64)
    atom_types = onp.array(lammps_step_0[:, 1], dtype=onp.int32)

    scaled_positions = onp.array(lammps_step_0[:, 2:5], dtype=onp.float32)
    positions_np = scaled_positions * box_np[None, :]
    positions = jnp.array(positions_np, dtype=DTYPE)
    velocity = jnp.array(lammps_step_0[:, 5:8], dtype=DTYPE)

    N_real = int(positions.shape[0])

    type_to_Z = {1: 14}
    z_real = onp.array([type_to_Z[int(t)] for t in atom_types], dtype=onp.int32)

    print("Unique LAMMPS atom types:", onp.unique(atom_types))
    print("Unique atomic numbers z_real:", onp.unique(z_real))
    print("N atoms:", N_real)

    torch_device = configure_torch_runtime(get_torch_device(), deterministic=True)

    torch_model = load_foundation_model(
        foundation_source,
        foundation_variant,
        device="cpu",
    )
    torch_model_config = extract_config_mace_model(torch_model)

    model_Z = set(int(x) for x in torch_model_config["atomic_numbers"])
    sys_Z = set(int(x) for x in onp.unique(z_real))
    print("Model supports Z:", sorted(model_Z))
    print("System has Z:", sorted(sys_Z))
    print("Unsupported Z in system:", sorted(sys_Z - model_Z))

    if sys_Z - model_Z:
        raise ValueError(
            f"Model does not support atomic numbers: {sorted(sys_Z - model_Z)}"
        )

    torch_model = torch_model.to(torch_device)

    graphdef, nnx_state, jax_model_config = convert_model(
        torch_model,
        torch_model_config,
    )
    jax_model = nnx.merge(graphdef, nnx_state)

    unit = units.metal_unit_system()
    dt = jnp.asarray(timestep_ps * unit["time"], dtype=DTYPE)
    T_init = jnp.asarray(300.0 * unit["temperature"], dtype=DTYPE)
    mass = jnp.asarray(28.0855 * unit["mass"], dtype=DTYPE)
    key = random.PRNGKey(121)

    nsave = nsteps_sim // write_every
    r_cutoff = float(jax_model_config["r_max"])

    displacement_fn, shift_fn = space.periodic_general(
        box_matrix,
        fractional_coordinates=False,
    )

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        format=partition.Dense,
    )

    featurize = mace_featurizer(
        displacement_fn,
        jax_model_config,
        z_real,
        fractional_coordinates=False,
    )

    def _energy_from_batch(batch):
        out = jax_model(batch, compute_stress=False)
        if isinstance(out, dict):
            if "energy" in out:
                return jnp.sum(out["energy"])
            if "energies" in out:
                return jnp.sum(out["energies"])
            raise KeyError(f"Could not find energy key in model output: {out.keys()}")
        return jnp.sum(out)

    @jit
    def energy_fn(R, box=None, neighbor=None, neighbors=None, **kwargs):
        del kwargs
        if neighbor is None:
            neighbor = neighbors
        if neighbor is None:
            raise ValueError("Provide neighbor=... or neighbors=...")

        if box is None:
            box_ = box_matrix
        else:
            box_ = jnp.asarray(box)
            if box_.shape == (3,):
                box_ = jnp.diag(box_)

        batch = featurize(R, neighbor, box=box_)
        return _energy_from_batch(batch)

    @jit
    def pressure_snapshot_fn(state, box, neighbor):
        box_vec = jnp.asarray(box, dtype=DTYPE)
        box_mat = jnp.diag(box_vec) if box_vec.shape == (3,) else box_vec
        volume = jnp.linalg.det(box_mat)

        kinetic = quantity.kinetic_energy(momentum=state.momentum, mass=mass)

        def scaled_energy(eps):
            perturbation = jnp.eye(3, dtype=DTYPE) * (1.0 + eps)
            batch = featurize(
                state.position,
                neighbor,
                box=box_mat,
                perturbation=perturbation,
            )
            return _energy_from_batch(batch)

        dU_deps = grad(scaled_energy)(jnp.asarray(0.0, dtype=DTYPE))
        pressure = (2.0 * kinetic - dU_deps) / (3.0 * volume)
        return pressure

    frame0 = read_lammps_dump_timestep(dump_file, target_timestep=0)

    pos_dump = frame0["pos"].astype(onp.float32)
    vel_dump = frame0["vel"].astype(onp.float32)
    force_lmp = frame0["force"].astype(onp.float32)
    types_dump = frame0["types"].astype(onp.int32)

    if not onp.array_equal(types_dump, atom_types):
        raise ValueError("LAMMPS dump atom types do not match initialization ordering.")

    positions_dump = jnp.array(pos_dump, dtype=DTYPE)
    velocity_dump = jnp.array(vel_dump, dtype=DTYPE)

    nbrs_dump = neighbor_fn.allocate(positions_dump, extra_capacity=2)

    print("Dump neighbor overflow:", bool(nbrs_dump.did_buffer_overflow))
    print("Dump neighbor list shape:", nbrs_dump.idx.shape)

    E0 = energy_fn(positions_dump, box=box, neighbor=nbrs_dump)
    print(f"Initial energy on dump timestep 0: {float(E0):.8f} eV")

    force_fn = jit(
        lambda R, nbrs_: -grad(lambda X: energy_fn(X, box=box, neighbor=nbrs_))(R)
    )
    F0 = force_fn(positions_dump, nbrs_dump)

    F_jax = onp.array(F0, dtype=onp.float64)
    diff = F_jax - force_lmp.astype(onp.float64)
    err = onp.linalg.norm(diff, axis=1)

    print("Force comparison on dump timestep 0:")
    print("  max error:", float(err.max()))
    print("  mean error:", float(err.mean()))
    print("  rms error:", float(onp.sqrt(onp.mean(err**2))))

    abs_diff = onp.abs(diff)
    print("Component-wise max abs error:")
    print("  fx:", float(abs_diff[:, 0].max()))
    print("  fy:", float(abs_diff[:, 1].max()))
    print("  fz:", float(abs_diff[:, 2].max()))

    print("Sum of LAMMPS forces:", force_lmp.sum(axis=0))
    print("Sum of JAX forces:   ", F_jax.sum(axis=0))
    print("Sum of force diff:   ", diff.sum(axis=0))

    state0 = dataclasses.replace(
        simulate.nve(energy_fn, shift_fn, dt=dt)[0](
            key,
            positions_dump,
            box=box,
            neighbor=nbrs_dump,
            kT=T_init,
            mass=mass,
        ),
        momentum=mass * velocity_dump * unit["velocity"],
    )

    P0 = pressure_snapshot_fn(state0, box, nbrs_dump)

    print("JAX pressure at dump timestep 0 (bar):", float(P0 / unit["pressure"]))
    print("LAMMPS pressure at timestep 0 (bar):", float(P_lammps[0]))

    worst = onp.argsort(err)[-10:][::-1]
    print("\nTop 10 worst force mismatches:")
    for idx in worst:
        print(
            f"atom {idx+1:4d} | "
            f"pos={pos_dump[idx]} | "
            f"F_lmp={force_lmp[idx]} | "
            f"F_jax={F_jax[idx]} | "
            f"dF={diff[idx]} | "
            f"|dF|={err[idx]:.8f}"
        )

    L = box_np
    dist_to_low = pos_dump
    dist_to_high = L[None, :] - pos_dump
    dist_to_boundary = onp.min(
        onp.concatenate([dist_to_low, dist_to_high], axis=1), axis=1
    )

    print(
        "Mean |dF| for atoms within 0.5 A of boundary:",
        float(err[dist_to_boundary < 0.5].mean()),
    )
    print(
        "Mean |dF| for atoms within 1.5 A of boundary:",
        float(err[dist_to_boundary < 1.5].mean()),
    )
    print(
        "Mean |dF| for atoms farther than 3.0 A from boundary:",
        float(err[dist_to_boundary > 3.0].mean()),
    )

    positions = positions_dump
    velocity = velocity_dump
    nbrs = nbrs_dump

    init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=dt)
    apply_fn = jit(apply_fn)

    state = init_fn(key, positions, box=box, neighbor=nbrs, kT=T_init, mass=mass)
    state = dataclasses.replace(state, momentum=mass * velocity * unit["velocity"])

    @jit
    def step_fn(i, state_nbrs):
        del i
        state_, nbrs_ = state_nbrs
        state_ = apply_fn(state_, neighbor=nbrs_)
        nbrs_ = nbrs_.update(state_.position)
        return state_, nbrs_

    pressure_every_blocks = 10
    log_every_blocks = pressure_every_blocks

    log_blocks = onp.arange(0, nsave, log_every_blocks, dtype=int)
    if log_blocks[-1] != nsave - 1:
        log_blocks = onp.append(log_blocks, nsave - 1)

    nlog = len(log_blocks)

    log = {
        "E_pot": jnp.zeros((nlog,), dtype=DTYPE),
        "E_kin": jnp.zeros((nlog,), dtype=DTYPE),
        "E_tot": jnp.zeros((nlog,), dtype=DTYPE),
        "Temp": jnp.zeros((nlog,), dtype=DTYPE),
        "P": jnp.zeros((nlog,), dtype=DTYPE),
        "block": jnp.zeros((nlog,), dtype=jnp.int32),
        "step": jnp.zeros((nlog,), dtype=jnp.int32),
    }

    ilog = 0

    for j in range(nsave):
        if (j % log_every_blocks == 0) or (j == nsave - 1):
            K = quantity.kinetic_energy(momentum=state.momentum, mass=mass)
            E = energy_fn(state.position, box=box, neighbor=nbrs)
            Temp = quantity.temperature(momentum=state.momentum, mass=mass)
            Etot = K + E
            P = pressure_snapshot_fn(state, box, nbrs)

            log["E_pot"] = log["E_pot"].at[ilog].set(E)
            log["E_kin"] = log["E_kin"].at[ilog].set(K)
            log["E_tot"] = log["E_tot"].at[ilog].set(Etot)
            log["Temp"] = log["Temp"].at[ilog].set(Temp)
            log["P"] = log["P"].at[ilog].set(P)
            log["block"] = log["block"].at[ilog].set(j)
            log["step"] = log["step"].at[ilog].set(j * write_every)

            if verbose:
                print(
                    f"Log {ilog + 1:5d}/{nlog:5d} | "
                    f"block={j:5d}/{nsave - 1:5d} | "
                    f"step={(j * write_every):7d} | "
                    f"E_pot={float(E): .8f} eV | "
                    f"E_kin={float(K): .8f} eV | "
                    f"E_tot={float(Etot): .8f} eV | "
                    f"T={float(Temp / unit['temperature']): .4f} K | "
                    f"P={float(P / unit['pressure']): .4f} bar"
                )

            ilog += 1

        state, nbrs = lax.fori_loop(0, write_every, step_fn, (state, nbrs))

    overflow = bool(nbrs.did_buffer_overflow)
    print(f"Neighbor buffer overflow: {overflow}")

    logged_steps = onp.array(log["step"])
    t = logged_steps * timestep_ps
    natoms = positions.shape[0]

    lammps_steps = data_lammps["Step"].to_numpy()
    common_steps = logged_steps[onp.isin(logged_steps, lammps_steps)]

    if len(common_steps) == 0:
        raise ValueError("No overlapping steps found between JAX log and LAMMPS thermo data.")

    step_to_row = pd.Series(onp.arange(len(lammps_steps)), index=lammps_steps)
    idx_lammps = step_to_row.loc[common_steps].to_numpy()
    idx_jax = onp.nonzero(onp.isin(logged_steps, common_steps))[0]

    t_plot = t[idx_jax]
    t_l_plot = t_l[idx_lammps]

    T_l_plot = T_lammps[idx_lammps]
    P_l_plot = P_lammps[idx_lammps]
    E_l_plot = E_lammps[idx_lammps]
    H_l_plot = H_lammps[idx_lammps]

    Temp_jax = onp.array(log["Temp"][idx_jax] / unit["temperature"])
    P_jax = onp.array((log["P"][idx_jax] / unit["pressure"]) / 10000.0)
    Epot_jax = onp.array(log["E_pot"][idx_jax])
    Ekin_jax = onp.array(log["E_kin"][idx_jax])
    Etot_jax = onp.array(log["E_tot"][idx_jax])
    Etot_per_atom_jax = Etot_jax / natoms

    log_df = pd.DataFrame(
        {
            "step": onp.array(log["step"]),
            "time_ps": onp.array(log["step"]) * timestep_ps,
            "temperature_K": onp.array(log["Temp"] / unit["temperature"]),
            "pressure_GPa": onp.array((log["P"] / unit["pressure"]) / 10000.0),
            "potential_energy_eV": onp.array(log["E_pot"]),
            "kinetic_energy_eV": onp.array(log["E_kin"]),
            "total_energy_eV": onp.array(log["E_tot"]),
            "total_energy_eV_per_atom": onp.array(log["E_tot"] / natoms),
        }
    )
    log_df.to_csv(output_dir / "jax_md_nve_log.csv", index=False)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams.update({"font.size": 12})

    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t_plot, Temp_jax, lw=3, label="JAX-MD + MACE")
    ax1.plot(t_l_plot, T_l_plot, lw=2, label="LAMMPS + MACE")
    ax1.set_title("Temperature")
    ax1.set_ylabel(r"$T\ (K)$")
    ax1.set_xlabel(r"$t\ (ps)$")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t_plot, P_jax, lw=3, label="JAX-MD + MACE")
    ax2.plot(t_l_plot, P_l_plot / 10000.0, lw=2, label="LAMMPS + MACE")
    ax2.set_title("Pressure")
    ax2.set_ylabel(r"$P\ (GPa)$")
    ax2.set_xlabel(r"$t\ (ps)$")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t_plot, Epot_jax, lw=3, label="JAX-MD + MACE")
    ax3.plot(t_l_plot, E_l_plot, lw=2, label="LAMMPS + MACE")
    ax3.set_title("Potential Energy")
    ax3.set_ylabel(r"$E_{PE}\ (eV)$")
    ax3.set_xlabel(r"$t\ (ps)$")
    ax3.legend()

    mean_et = Etot_per_atom_jax.mean()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t_plot, Etot_per_atom_jax, lw=3, label="JAX-MD + MACE")
    ax4.plot(t_l_plot, H_l_plot / natoms, lw=2, label="LAMMPS + MACE")
    ax4.set_title("Constant of Motion")
    ax4.set_ylabel(r"$E_T\ (eV/atom)$")
    ax4.set_xlabel(r"$t\ (ps)$")
    ax4.set_ylim(mean_et - abs(mean_et) / 1000.0, mean_et + abs(mean_et) / 1000.0)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    nskip = 1
    jax_energy = onp.array(log["E_pot"][nskip:] / natoms)

    if len(jax_energy) >= 2:
        kde_jax = stats.gaussian_kde(jax_energy)
        x_range = onp.linspace(jax_energy.min(), jax_energy.max(), 200)

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(x_range, kde_jax(x_range), linewidth=3, label="JAX-MD + MACE", alpha=0.8)

        lammps_energy = onp.array(E_l_plot[nskip:] / natoms)
        if len(lammps_energy) >= 2:
            kde_lammps = stats.gaussian_kde(lammps_energy)
            x_range_lammps = onp.linspace(lammps_energy.min(), lammps_energy.max(), 200)

            plt.plot(
                x_range_lammps,
                kde_lammps(x_range_lammps),
                linewidth=3,
                label="LAMMPS + MACE",
                linestyle="--",
                alpha=0.8,
            )

        plt.xlabel("Potential Energy (eV/atom)")
        plt.ylabel("Probability Density")
        plt.title("Energy Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(output_dir / "energy_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

    print(f"Results written to: {output_dir.resolve()}")
    print("Generated files:")
    print(f"  - {output_dir / 'jax_md_nve_log.csv'}")
    print(f"  - {output_dir / 'comparison_plot.png'}")
    print(f"  - {output_dir / 'energy_distribution.png'}")


if __name__ == "__main__":
    main()
