#!/usr/bin/env python3

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config_mod

jax_config_mod.update("jax_enable_x64", True)

from jax_md import simulate, units, quantity
from jax_md.dataclasses import replace as dc_replace

from extract_params_oplsaa import parse_lammps_data

from mace_setup import build_mace_model
from mace_energy import build_mace_energy_system
from md_drivers import MDConfig
from lammps_io import (
    read_lammps_system,
    read_lammps_velocities,
    write_lammps_dump_frame,
)

def uses_dynamic_box(cfg):
    return cfg.ensemble == "npt"


def allocate_neighbors(neighbor_fn, R, box, cfg):
    if uses_dynamic_box(cfg):
        return neighbor_fn.allocate(
            R,
            box=box,
            extra_capacity=2,
        )

    return neighbor_fn.allocate(
        R,
        extra_capacity=2,
    )


def update_neighbors(nbrs, R, box, cfg):
    if uses_dynamic_box(cfg):
        return nbrs.update(
            R,
            box=box,
        )

    return nbrs.update(R)


def make_box_matrix(box):
    box = np.asarray(box)
    if box.ndim == 1:
        return jnp.diag(jnp.asarray(box))
    if box.ndim == 2:
        return jnp.asarray(box)
    raise ValueError(f"Unexpected box shape: {box.shape}")


def remove_com_momentum_fn(mass_vec):
    @jax.jit
    def remove_com_momentum(state):
        total_p = jnp.sum(state.momentum, axis=0)
        total_m = jnp.sum(mass_vec)
        v_cm = total_p / total_m
        return dc_replace(state, momentum=state.momentum - mass_vec[:, None] * v_cm)

    return remove_com_momentum


def neighbor_cap_stats(nbrs_idx, n_atoms, k):
    idx = nbrs_idx
    valid = (idx >= 0) & (idx < n_atoms)

    self_edge = idx == jnp.arange(n_atoms, dtype=idx.dtype)[:, None]
    valid = valid & (~self_edge)

    counts = jnp.sum(valid, axis=1)
    dropped = jnp.maximum(0, counts - k)

    return (
        jnp.max(counts),
        jnp.mean(counts >= k),
        jnp.mean(counts > k),
        jnp.mean(dropped),
        jnp.max(dropped),
    )


def main():
    total_time_start = time.time()

    cfg = MDConfig(
        ensemble="nve",
        dt_ps=0.001,
        temperature_K=298.0,
        n_steps=250000,
        print_every=1000,
    )

    data_file = "EC_data_in.data"
    settings_file = "EC.settings"
    velocity_file = "dump_EC_velo.lammpstrj"
    dump_filename = "dump_mixture_nve_unwrapped.lammpstrj"

    update_interval = 1
    k_neighbors = 96

    type_to_Z = {
        1: 6,  # C
        2: 8,  # O
        3: 1,  # H
    }

    system = read_lammps_system(
        data_file,
        settings_file,
        type_to_Z,
        parse_lammps_data,
    )

    positions = np.asarray(system["positions"], dtype=np.float64)
    box = np.asarray(system["box"], dtype=np.float64)
    masses = np.asarray(system["masses"], dtype=np.float64)
    atom_types = system["atom_types"]
    z_atomic = system["z_atomic"]
    n_atoms = system["n_atoms"]

    velocities = read_lammps_velocities(velocity_file, n_atoms)

    print("box raw:", box)
    print("N atoms:", n_atoms)
    print("unique LAMMPS atom types:", np.unique(atom_types))
    print("unique atomic numbers:", np.unique(z_atomic))

    box0 = make_box_matrix(box)
    positions_cart = jnp.asarray(positions, dtype=jnp.float64)
    velocities_cart = jnp.asarray(velocities, dtype=jnp.float64)

    jax_model, jax_model_config, torch_config = build_mace_model(
        source="mp",
        variant="small-0b2",
    )

    model_Z = set(int(x) for x in torch_config["atomic_numbers"])
    system_Z = set(int(x) for x in np.unique(z_atomic))
    unsupported = sorted(system_Z - model_Z)

    print("Model supports Z:", sorted(model_Z))
    print("System has Z:", sorted(system_Z))
    print("Unsupported Z:", unsupported)

    if unsupported:
        raise ValueError(f"MACE model does not support atomic numbers: {unsupported}")

    displacement_fn, shift_fn, neighbor_fn, make_energy_fn = build_mace_energy_system(
        jax_model=jax_model,
        jax_model_config=jax_model_config,
        z_atomic=z_atomic,
        box0=box0,
        ensemble=cfg.ensemble,
        dr_threshold=0.5,
        capacity_multiplier=1.0,
    )

    unit = units.metal_unit_system()

    dt = jnp.asarray(cfg.dt_ps * unit["time"], dtype=jnp.float64)
    kT = jnp.asarray(cfg.temperature_K * unit["temperature"], dtype=jnp.float64)

    mass_vec = jnp.asarray(masses, dtype=jnp.float64) * unit["mass"]
    mass_mat = mass_vec[:, None]

    key = jax.random.PRNGKey(121)

    nbrs = allocate_neighbors(
     neighbor_fn,
     positions_cart,
     box0,
     cfg,
    )

    print("Neighbor idx shape:", nbrs.idx.shape)
    print("Neighbor slots total:", np.prod(np.array(nbrs.idx.shape)))
    nbrs = update_neighbors(
     nbrs,
     positions_cart,
     box0,
     cfg,
    )

    print("Neighbor list slot capacity per atom:", int(nbrs.idx.shape[1]))

    energy_fn = make_energy_fn(nbrs)

    def energy_nve(R, *, neighbor):
        return make_energy_fn(neighbor)(R, box=box0)

    init_fn, apply_fn = simulate.nve(
        energy_nve,
        shift_fn,
        dt=dt,
        mass=mass_mat,
    )

    state = init_fn(
        key,
        positions_cart,
        kT=kT,
        neighbor=nbrs,
    )

    state = dc_replace(
        state,
        mass=mass_mat,
        momentum=mass_mat * velocities_cart * unit["velocity"],
    )

    remove_com_momentum = remove_com_momentum_fn(mass_vec)
    state = remove_com_momentum(state)

    unwrapped_position = state.position

    kinetic_energy = quantity.kinetic_energy
    temperature_fn = quantity.temperature

    PE0 = energy_fn(state.position, box=box0)
    KE0 = kinetic_energy(momentum=state.momentum, mass=mass_mat)
    T0 = temperature_fn(momentum=state.momentum, mass=mass_mat) / unit["temperature"]

    print(
        "t=0 KE, PE, E, T:",
        float(KE0),
        float(PE0),
        float(KE0 + PE0),
        float(T0),
    )


    @jax.jit
    def step_once(state, nbrs):
        state = apply_fn(state, neighbor=nbrs)

        if cfg.ensemble == "npt":
           box_now = simulate.npt_box(state)

           nbrs = nbrs.update(
             state.position,
             box=box_now,
           )
        else:
           nbrs = nbrs.update(state.position)

        return state, nbrs

    atom_ids = np.arange(1, n_atoms + 1)
    dump_file = open(dump_filename, "w")

    print("step\tKE\t\tPE\t\tE\t\tT(K)\tsec/step")
    t0 = time.time()

    for step in range(cfg.n_steps):
        old_pos = state.position

        do_update = step % update_interval == 0
        state, nbrs = step_once(state, nbrs)
        state = remove_com_momentum(state)

        dR = jax.vmap(displacement_fn)(state.position, old_pos)
        unwrapped_position = unwrapped_position + dR

        if step % cfg.print_every == 0:
            energy_fn = make_energy_fn(nbrs)

            KE = kinetic_energy(momentum=state.momentum, mass=mass_mat)
            PE = energy_fn(state.position, box=box0)
            T_inst = temperature_fn(
                momentum=state.momentum,
                mass=mass_mat,
            ) / unit["temperature"]

            sec_per_step = (time.time() - t0) / max(1, step + 1)

            print(
                f"{step}\t"
                f"{float(KE):.2f}\t"
                f"{float(PE):.2f}\t"
                f"{float(KE + PE):.3f}\t"
                f"{float(T_inst):.1f}\t"
                f"{sec_per_step:.4f}"
            )

            maxc, frac_ge_k, frac_gt_k, mean_drop, max_drop = neighbor_cap_stats(
                nbrs.idx,
                n_atoms,
                k_neighbors,
            )

            print(
                f"nbr stats: "
                f"max={float(maxc):.0f} "
                f"frac>=k={float(frac_ge_k):.3f} "
                f"mean_drop={float(mean_drop):.3f} "
                f"max_drop={float(max_drop):.0f}"
            )

            write_lammps_dump_frame(
                file=dump_file,
                step=step,
                box=box,
                atom_ids=atom_ids,
                atom_types=atom_types,
                masses=masses,
                wrapped_positions=np.array(state.position),
                unwrapped_positions=np.array(unwrapped_position),
            )

    dump_file.close()

    print("Total wall time (s):", time.time() - total_time_start)
    jax.clear_caches()


if __name__ == "__main__":
    main()
