"""Planar two-joint human arm moved by six muscles, in JAX (differentiable, batched).

Rigid-body and muscle geometry follow Li & Todorov (2004): shoulder and elbow move in the horizontal
plane (no gravity), six muscles with constant moment arms (shoulder flexor/extensor, elbow
flexor/extensor, biarticular flexor/extensor), first-order activation dynamics, and joint viscosity.
Muscle force follows a Hill-type force-velocity relation, so active muscle resists stretch: the arm's
damping grows with activation and the controller can stiffen it by co-contracting.
Sensory feedback is delayed (proprioception 30 ms, vision 70 ms) and noisy; motor commands carry
signal-dependent noise (Harris & Wolpert 1998).

Forces on the hand: a velocity-dependent curl field (Shadmehr & Mussa-Ivaldi 1994) and/or a force
channel that clamps the hand to the straight start-target line (Scheidt et al. 2000).  Both are set
per call through `Env`, so one compiled rollout serves every experimental condition.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

DT = 0.01               # controller period [s]
SUBSTEPS = 2            # physics steps per controller step
L1, L2 = 0.30, 0.33     # upper arm, forearm length [m]
M1, M2 = 1.4, 1.0       # segment masses [kg]
S1, S2 = 0.11, 0.16     # joint-to-centre-of-mass distances [m]
I1, I2 = 0.025, 0.045   # inertias about the centres of mass [kg m^2]
VISCOSITY = jnp.array([[0.05, 0.025], [0.025, 0.05]])              # joint viscosity [N m s/rad]
MOMENT_ARMS = jnp.array([[0.020, -0.020, 0.000, 0.000, 0.015, -0.020],   # shoulder [m]
                         [0.000, 0.000, 0.020, -0.020, 0.020, -0.015]])  # elbow
MUSCLES = ("shoulder flexor", "shoulder extensor", "elbow flexor", "elbow extensor",
           "biarticular flexor", "biarticular extensor")
F_MAX = jnp.array([800.0, 800.0, 1000.0, 1000.0, 600.0, 600.0])    # max isometric force [N]
TAU_ACT, TAU_DEACT = 0.015, 0.050                                    # activation time constants [s]
V_MAX = 1.0                                                          # max fibre velocity (~10 optimal lengths/s) [m/s]
Q_LO, Q_HI = jnp.array([-0.6, 0.05]), jnp.array([2.6, 2.7])          # soft joint limits [rad]
Q_CENTER = jnp.array([jnp.pi / 4, jnp.pi / 2])                       # posture for centre-out reaches
DELAY_PROPRIO, DELAY_VISION = 3, 7                                   # feedback delays [controller steps]
NOISE_PROPRIO, NOISE_VISION, NOISE_MOTOR = 0.02, 0.001, 0.05         # spindle (normalised), vision [m], SDN
CURL = 13.0                                                          # curl-field gain [N s/m]
CHANNEL_K, CHANNEL_B = 2000.0, 50.0                                  # channel wall stiffness, damping

N_OBS = 2 + 1 + 2 + 12   # target, go cue, seen hand position, spindle lengths + velocities


class Env(NamedTuple):
    curl: float = 0.0      # curl-field gain [N s/m]; CURL for the force field
    channel: float = 0.0   # 1.0 clamps the hand to the start-target line
    noise: float = 1.0     # scales all sensory and motor noise


class ArmState(NamedTuple):
    q: jax.Array          # joint angles (B, 2)
    qd: jax.Array         # joint velocities (B, 2)
    act: jax.Array        # muscle activations (B, 6)
    proprio: jax.Array    # spindle signals still in transit (DELAY_PROPRIO + 1, B, 12)
    vision: jax.Array     # hand positions still in transit (DELAY_VISION + 1, B, 2)


def hand_position(q):
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    return jnp.stack([L1 * jnp.cos(q1) + L2 * jnp.cos(q12), L1 * jnp.sin(q1) + L2 * jnp.sin(q12)], -1)


def hand_jacobian(q):
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    s1, s12, c1, c12 = jnp.sin(q1), jnp.sin(q12), jnp.cos(q1), jnp.cos(q12)
    return jnp.stack([jnp.stack([-L1 * s1 - L2 * s12, -L2 * s12], -1),
                      jnp.stack([L1 * c1 + L2 * c12, L2 * c12], -1)], -2)


def hand_velocity(q, qd):
    return jnp.einsum("bij,bj->bi", hand_jacobian(q), qd)


def force_velocity(v_lengthening):
    """Hill-type force-velocity factor: 0 at fast shortening, 1 isometric, 1.8 at fast lengthening (slope 5/V_MAX)."""
    return 0.9 + 0.9 * jnp.tanh(5.62 * v_lengthening / V_MAX + 0.1115)


def spindles(q, qd):
    """Muscle length and velocity changes (linear in joint motion for constant moment arms), normalised."""
    return jnp.concatenate([-(q - Q_CENTER) @ MOMENT_ARMS / 0.02, -qd @ MOMENT_ARMS / 0.02 * 0.2], -1)


def reset(q0):
    qd = jnp.zeros_like(q0)
    return ArmState(q0, qd, jnp.zeros((q0.shape[0], 6)),
                    jnp.repeat(spindles(q0, qd)[None], DELAY_PROPRIO + 1, 0),
                    jnp.repeat(hand_position(q0)[None], DELAY_VISION + 1, 0))


def observe(state, target_seen, go_seen, env, noise):
    """Delayed, noisy senses.  `noise` holds 14 standard normals per arm (12 spindle, 2 vision)."""
    proprio = state.proprio[0] + env.noise * NOISE_PROPRIO * noise[:, :12]
    vision = state.vision[0] + env.noise * NOISE_VISION * noise[:, 12:]
    return jnp.concatenate([target_seen * 5, go_seen, vision * 5, proprio], -1)


def external_force(x, v, start, target, env):
    curl = env.curl * jnp.stack([v[:, 1], -v[:, 0]], -1)
    direction = (target - start) / jnp.linalg.norm(target - start, axis=-1, keepdims=True)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
    d_perp = jnp.sum((x - start) * normal, -1, keepdims=True)
    v_perp = jnp.sum(v * normal, -1, keepdims=True)
    channel = -(CHANNEL_K * d_perp + CHANNEL_B * v_perp) * normal
    return curl + env.channel * channel, env.channel * channel


def step(state, excitation, start, target, env, noise, f_pulse):
    """Advance one controller period (`noise`: 6 standard normals per arm for signal-dependent motor noise;
    `f_pulse`: an extra force on the hand, used for random perturbations during training).

    Returns the new state and the channel wall force on the hand.
    """
    excitation = excitation * (1 + env.noise * NOISE_MOTOR * noise)
    excitation = jnp.clip(excitation, 0.0, 1.0)
    tau = jnp.where(excitation > state.act, TAU_ACT, TAU_DEACT)
    act = state.act + (1 - jnp.exp(-DT / tau)) * (excitation - state.act)

    a1 = I1 + I2 + M1 * S1 ** 2 + M2 * (L1 ** 2 + S2 ** 2)
    a2 = M2 * L1 * S2
    a3 = I2 + M2 * S2 ** 2
    h = DT / SUBSTEPS
    q, qd = state.q, state.qd
    wall = jnp.zeros_like(q)
    for _ in range(SUBSTEPS):
        c2, s2 = jnp.cos(q[:, 1]), jnp.sin(q[:, 1])
        m11, m12, m22 = a1 + 2 * a2 * c2, a3 + a2 * c2, a3
        coriolis = jnp.stack([-a2 * s2 * qd[:, 1] * (2 * qd[:, 0] + qd[:, 1]), a2 * s2 * qd[:, 0] ** 2], -1)
        J = hand_jacobian(q)
        f_ext, f_wall = external_force(hand_position(q), jnp.einsum("bij,bj->bi", J, qd), start, target, env)
        f_ext = f_ext + f_pulse
        wall = wall + f_wall / SUBSTEPS
        torque_muscle = (act * F_MAX * force_velocity(-qd @ MOMENT_ARMS)) @ MOMENT_ARMS.T
        limits = 50.0 * (jnp.clip(Q_LO - q, 0) - jnp.clip(q - Q_HI, 0))
        tau = torque_muscle - coriolis - qd @ VISCOSITY.T + limits + jnp.einsum("bji,bj->bi", J, f_ext)
        det = m11 * m22 - m12 ** 2
        qdd = jnp.stack([m22 * tau[:, 0] - m12 * tau[:, 1], -m12 * tau[:, 0] + m11 * tau[:, 1]], -1) / det[:, None]
        qd = qd + h * qdd                      # semi-implicit Euler
        q = q + h * qd

    state = ArmState(q, qd, act,
                     jnp.concatenate([state.proprio[1:], spindles(q, qd)[None]]),
                     jnp.concatenate([state.vision[1:], hand_position(q)[None]]))
    return state, wall
