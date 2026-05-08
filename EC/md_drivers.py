from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import random
from jax_md import simulate, quantity, units, dataclasses as jmd_dataclasses


Ensemble = Literal["nve", "nvt", "npt"]


@dataclass
class MDConfig:
    ensemble: Ensemble = "nve"
    dt_ps: float = 1e-3
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    tau_t_steps: float = 100.0
    tau_p_steps: float = 1000.0
    n_steps: int = 1000
    print_every: int = 100


def current_box(cfg, state, box0):
    return simulate.npt_box(state) if cfg.ensemble == "npt" else box0


def make_apply_fn(cfg, energy_fn, shift_fn, dt, kT, pressure, tau_t, tau_p):
    if cfg.ensemble == "nve":
        return simulate.nve(energy_fn, shift_fn, dt=dt)

    if cfg.ensemble == "nvt":
        return simulate.nvt_nose_hoover(
            energy_fn,
            shift_fn,
            dt=dt,
            kT=kT,
            thermostat_kwargs={"tau": tau_t},
        )

    if cfg.ensemble == "npt":
        return simulate.npt_nose_hoover(
            energy_fn,
            shift_fn,
            dt=dt,
            pressure=pressure,
            kT=kT,
            thermostat_kwargs={"tau": tau_t},
            barostat_kwargs={"tau": tau_p},
        )

    raise ValueError(f"Unknown ensemble: {cfg.ensemble!r}")


def run_md(
    *,
    cfg,
    positions_cart0,
    velocities_cart0,
    box0,
    mass_amu,
    shift_fn,
    neighbor_fn,
    make_energy_fn,
    seed=0,
):
    unit = units.metal_unit_system()

    dt = jnp.asarray(cfg.dt_ps * unit["time"])
    kT = jnp.asarray(cfg.temperature_K * unit["temperature"])
    pressure = jnp.asarray(cfg.pressure_bar * unit["pressure"])
    tau_t = dt * cfg.tau_t_steps
    tau_p = dt * cfg.tau_p_steps
    mass = jnp.asarray(mass_amu * unit["mass"])

    key = random.PRNGKey(seed)

    if cfg.ensemble == "npt":
        positions0 = positions_cart0 @ jnp.linalg.inv(box0)
    else:
        positions0 = positions_cart0

    nbrs = neighbor_fn.allocate(
        positions0,
        box=box0,
        extra_capacity=2,
    )

    energy_fn = make_energy_fn(nbrs)
    init_fn, apply_fn = make_apply_fn(
        cfg, energy_fn, shift_fn, dt, kT, pressure, tau_t, tau_p
    )

    state = init_fn(
        key,
        positions0,
        box=box0,
        mass=mass,
    )

    state = jmd_dataclasses.replace(
        state,
        momentum=mass * velocities_cart0 * unit["velocity"],
    )

    for step in range(1, cfg.n_steps + 1):
        box_now = current_box(cfg, state, box0)

        nbrs = nbrs.update(state.position, box=box_now)
        if bool(nbrs.did_buffer_overflow):
            nbrs = neighbor_fn.allocate(
                state.position,
                box=box_now,
                extra_capacity=2,
            )

        energy_fn = make_energy_fn(nbrs)
        _, apply_fn = make_apply_fn(
            cfg, energy_fn, shift_fn, dt, kT, pressure, tau_t, tau_p
        )

        state = apply_fn(state)

        if step % cfg.print_every == 0 or step == cfg.n_steps:
            box_now = current_box(cfg, state, box0)
            kinetic = quantity.kinetic_energy(state.momentum, mass)
            temperature = quantity.temperature(state.momentum, mass)
            potential = energy_fn(state.position, box=box_now)

            msg = (
                f"step={step:7d} | "
                f"E_pot={float(potential): .8f} eV | "
                f"E_kin={float(kinetic): .8f} eV | "
                f"T={float(temperature / unit['temperature']): .3f} K"
            )

            if cfg.ensemble == "npt":
                pressure_inst = quantity.pressure(
                    energy_fn,
                    state.position,
                    box_now,
                    kinetic_energy=kinetic,
                )
                volume = jnp.linalg.det(box_now)
                msg += (
                    f" | P={float(pressure_inst / unit['pressure']): .3f} bar"
                    f" | V={float(volume): .6f}"
                )

            print(msg)

    return state, nbrs
