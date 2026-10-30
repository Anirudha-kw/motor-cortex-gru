"""Franka Panda (7-DoF) in MuJoCo MJX as a differentiable, torque-controlled plant.

The Menagerie MJX Panda without its hand, with the position servos removed so the controller commands
joint torques directly (the real Panda's torque mode), gravity compensated (as Franka's controller
does), and Menagerie's standard joint damping of 1 N m s/rad.  Collisions and constraint solving are
off, so the dynamics are smooth and can be differentiated end to end.  The "hand" is a handle point
10 cm past the flange; forces from the environment (curl field, force channel, perturbation pulses)
act there.
"""

import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

PANDA_XML = Path(os.environ.get("PANDA_XML", Path.home() / "mujoco_menagerie/franka_emika_panda/mjx_panda_nohand.xml"))
N_JOINTS = 7
DT = 0.01                                                    # controller period [s]
SUBSTEPS = 2                                                 # physics steps per controller step
HANDLE = 0.107 + 0.10                                        # flange + handle, along link7's z axis [m]
TAU_MAX = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])   # Panda joint torque limits [N m]
TAU_MOTOR = 0.01                                             # torque-loop lag [s]
Q_HOME = jnp.array([0.0, -0.35, 0.0, -2.3, 0.0, 1.95, 0.785])     # handle in front, flange pointing down
DELAY = 1                                                    # sensing delay [controller steps]
NOISE_Q, NOISE_QD, NOISE_MOTOR = 1e-3, 0.01, 0.02            # joint angle [rad], velocity [rad/s], SDN
CHANNEL_K, CHANNEL_B = 6000.0, 150.0                          # channel wall stiffness [N/m], damping [N s/m]
LIMIT_K = 200.0                                              # joint-stop stiffness beyond the joint limits [N m/rad]
PAYLOAD_RADIUS = 0.03                                        # payloads are solid spheres held at the handle [m]
GRAVITY = 9.81
RATE_MAX = 1000.0                                            # Franka's commanded-torque rate limit [N m/s]
V_FRICTION = 0.05                                            # velocity scale of the smoothed Coulomb friction [rad/s]


class Dyn(NamedTuple):
    """Per-arm dynamics randomisation for sim-to-real training (s2r.py); None everywhere = the nominal plant.
    All fields are batched over arms."""
    friction: jax.Array      # (B, 7) Coulomb friction [N m], smoothed f tanh(qd / V_FRICTION)
    viscous: jax.Array       # (B, 7) joint damping [N m s/rad] (nominal 1)
    bias: jax.Array          # (B, 7) torque offset, e.g. a gravity-compensation error [N m]
    gain: jax.Array          # (B, 7) torque gain error (applied = gain x commanded)
    mass_scale: jax.Array    # (B, 7) scale of each link's mass and inertia
    tool: jax.Array          # (B,) mass of a known, gravity-compensated tool at the handle [kg]: inertia only
    lag: jax.Array           # (B,) or (B, 7) torque-loop time constant [s] (nominal TAU_MOTOR)
    delay: jax.Array         # (B,) sensing delay [controller steps], 1 or 2
    noise: jax.Array         # (B,) sensor-noise scale
    armature: jax.Array = None   # (B, 7) reflected motor inertia [kg m^2]; None = the model's 0.1 (Menagerie)
    ctrl_damping: jax.Array = None   # (B, 7) joint damping the deployment controller adds [N m s/rad]; None = 0


class Env(NamedTuple):
    curl: float = 0.0      # curl-field gain [N s/m] in the horizontal plane
    channel: float = 0.0   # 1.0 clamps the handle to the vertical plane through start and target
    noise: float = 1.0


def build_mj_model():
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    spec.option.timestep = DT / SUBSTEPS
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER          # damping is integrated implicitly
    spec.option.gravity = [0.0, 0.0, 0.0]                               # gravity compensated
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    spec.body("link7").add_site(name="handle", pos=[0, 0, HANDLE], size=[0.01] * 3)
    # Massless placeholder at the handle; payload episodes set its mass and inertia per arm (see _advance).
    payload = spec.body("link7").add_body(name="payload", pos=[0, 0, HANDLE])
    payload.mass, payload.inertia, payload.explicitinertial = 1e-9, [1e-12] * 3, True
    model = spec.compile()
    model.dof_damping[:] = 1.0      # set on the compiled model: the spec field's type differs across versions
    return model


MJ_MODEL = build_mj_model()
MX = mjx.put_model(MJ_MODEL)
HANDLE_SITE = mujoco.mj_name2id(MJ_MODEL, mujoco.mjtObj.mjOBJ_SITE, "handle")
LINK7 = mujoco.mj_name2id(MJ_MODEL, mujoco.mjtObj.mjOBJ_BODY, "link7")
PAYLOAD = mujoco.mj_name2id(MJ_MODEL, mujoco.mjtObj.mjOBJ_BODY, "payload")
LINKS = jnp.array([mujoco.mj_name2id(MJ_MODEL, mujoco.mjtObj.mjOBJ_BODY, f"link{i}") for i in range(1, 8)])
Q_LO = jnp.array(MJ_MODEL.jnt_range[:, 0])
Q_HI = jnp.array(MJ_MODEL.jnt_range[:, 1])


class RobotState(NamedTuple):
    d: mjx.Data            # physics state (batched)
    tau: jax.Array         # torque actually applied after the motor lag (B, 7)
    sensed: jax.Array      # sensor readings in transit (buffer, B, 17): handle xyz, q, qd; [0] is the oldest
    cmd: jax.Array = None  # sim-to-real only: the controller's capped, rate-limited command (B, 7)


def _forward(q, qd):
    d = mjx.make_data(MX).replace(qpos=q, qvel=qd)
    return mjx.forward(MX, d)


def handle_position(d):
    return d.site_xpos[..., HANDLE_SITE, :]


def handle_jacobian(d):
    """3x7 Jacobian of the handle point, from the kinematics of the last forward pass."""
    jacp, _ = mjx.jac(MX, d, d.site_xpos[HANDLE_SITE], LINK7)
    return jacp.T


def handle_kinematics(q):
    """Handle position for joint angles q (batched)."""
    return jax.vmap(lambda q: handle_position(_forward(q, jnp.zeros(N_JOINTS))))(q)


def _sense(d):
    return jnp.concatenate([handle_position(d), d.qpos, d.qvel], -1)


def reset(q0, buffer=DELAY + 1, s2r=False):
    d = jax.vmap(_forward)(q0, jnp.zeros_like(q0))
    return RobotState(d, jnp.zeros_like(q0), jnp.repeat(_sense(d)[None], buffer, 0),
                      jnp.zeros_like(q0) if s2r else None)


def observe(state, home_handle, target_seen, go_seen, env, noise):
    """Delayed, noisy proprioception: handle position (from joint encoders + kinematics), q, qd."""
    s = state.sensed[0]
    handle = s[:, :3] + env.noise * NOISE_Q * 0.5 * noise[:, :3]
    q = s[:, 3:10] + env.noise * NOISE_Q * noise[:, 3:10]
    qd = s[:, 10:17] + env.noise * NOISE_QD * noise[:, 10:17]
    return jnp.concatenate([(target_seen - home_handle) * 5 * (target_seen != 0).any(-1, keepdims=True), go_seen,
                            (handle - home_handle) * 5, q - Q_HOME, qd * 0.2], -1)


def _external_force(p, v, start, target, env, f_pulse):
    curl = env.curl * jnp.stack([v[1], -v[0], 0.0])
    direction = (target - start).at[2].set(0.0)
    direction = direction / jnp.linalg.norm(direction)
    normal = jnp.array([-direction[1], direction[0], 0.0])
    wall = -(CHANNEL_K * jnp.dot(p - start, normal) + CHANNEL_B * jnp.dot(v, normal)) * normal * env.channel
    return curl + wall + f_pulse, wall


def _with_payload(mass):
    """The model with a solid-sphere payload of this mass at the handle (exact rigid-body dynamics in MJX)."""
    mass = jnp.maximum(mass, 1e-9)
    inertia = 0.4 * mass * PAYLOAD_RADIUS ** 2 * jnp.ones(3)
    return MX.replace(body_mass=MX.body_mass.at[PAYLOAD].set(mass), body_inertia=MX.body_inertia.at[PAYLOAD].set(inertia))


def _randomised(mass, dyn, qd):
    """One arm's model under a dynamics draw: payload + tool inertia at the handle, scaled links, and joint
    damping plus the smoothed Coulomb friction written as a velocity-dependent damping f tanh(qd/v)/qd, which
    MuJoCo's Euler step integrates implicitly (stable however stiff the friction is near zero velocity)."""
    model = _with_payload(mass + dyn.tool)
    scale = dyn.mass_scale
    small = jnp.abs(qd) < 1e-6
    ratio = jnp.where(small, 1.0 / V_FRICTION, jnp.tanh(qd / V_FRICTION) / jnp.where(small, 1.0, qd))
    model = model.replace(body_mass=model.body_mass.at[LINKS].multiply(scale),
                          body_inertia=model.body_inertia.at[LINKS].multiply(scale[:, None]),
                          dof_damping=dyn.viscous + dyn.friction * ratio)
    return model if dyn.armature is None else model.replace(dof_armature=dyn.armature)


def _advance(state, command, noise_scale, noise, force, force_args, payload=None, dyn=None, cap=None):
    """One controller period: torque command (B, 7) through the motor lag, then SUBSTEPS physics steps.

    force(p, v, *args) -> (force on the handle, channel-wall part of it), for one arm; force_args are batched.
    payload: optional per-arm payload mass (B,) [kg]; its weight must be part of force (gravity is off).
    dyn (Dyn) and cap (7,) [N m]: sim-to-real plant -- the controller caps the command and limits its rate to
    RATE_MAX, the torque loop lags with a per-arm or per-joint time constant, the robot controller may add
    joint damping (all three advanced every substep), the joints apply gain x torque + bias against
    friction, the links' masses and the delay are per arm.
    """
    command = command * (1 + noise_scale * NOISE_MOTOR * noise)
    cmd = state.cmd
    if dyn is None:
        tau = state.tau + (1 - jnp.exp(-DT / TAU_MOTOR)) * (command - state.tau)
    else:
        # the rate limit, the controller's damping and the torque lag are advanced every physics substep below:
        # updating them once per 10 ms period made the delivered torque arrive ~5 ms early, and the plant more
        # forgiving than the robot's 1 kHz loop (robot/tests/plant_probe.py "coarse")
        target0 = command if cap is None else jnp.clip(command, -cap, cap)
        alpha = 1 - jnp.exp(-(DT / SUBSTEPS) / dyn.lag)
        alpha = alpha[:, None] if alpha.ndim == 1 else alpha
        tau = state.tau

    def substep(d, tau, mass, *args):
        if dyn is not None:
            *args, arm = args
        J = handle_jacobian(d)
        f, wall = force(handle_position(d), J @ d.qvel, *args)
        stops = LIMIT_K * (jnp.clip(Q_LO - d.qpos, 0) - jnp.clip(d.qpos - Q_HI, 0))   # constraints are off
        if dyn is None:
            model, applied = (MX if mass is None else _with_payload(mass)), tau
        else:
            model, applied = _randomised(mass, arm, d.qvel), arm.gain * tau + arm.bias
        d = mjx.step(model, d.replace(qfrc_applied=applied + J.T @ f + stops))
        return d, wall

    d = state.d
    wall = jnp.zeros((tau.shape[0], 3))
    extra = () if dyn is None else (dyn,)
    for _ in range(SUBSTEPS):
        if dyn is not None:                   # the robot controller: clip(clip(policy) - D dq), then rate limit, lag
            target = target0
            if dyn.ctrl_damping is not None:
                target = target - dyn.ctrl_damping * d.qvel[:, :N_JOINTS]
                target = target if cap is None else jnp.clip(target, -cap, cap)
            cmd = cmd + jnp.clip(target - cmd, -RATE_MAX * DT / SUBSTEPS, RATE_MAX * DT / SUBSTEPS)
            tau = tau + alpha * (cmd - tau)
        d, w = jax.vmap(substep, in_axes=(0, 0, None if payload is None else 0) + (0,) * (len(force_args) + len(extra)))(
            d, tau, payload, *force_args, *extra)
        wall = wall + w / SUBSTEPS
    sensed = jnp.concatenate([state.sensed[1:], _sense(d)[None]])
    return RobotState(d, tau, sensed, cmd), wall


def step(state, command, start, target, env, noise, f_pulse):
    """One controller period in a curl field and/or force channel.  Returns the state and channel-wall force."""
    force = lambda p, v, start, target, f_pulse: _external_force(p, v, start, target, env, f_pulse)
    return _advance(state, command, env.noise, noise, force, (start, target, f_pulse))


def step_field(state, command, B, c, noise_scale, noise, f_pulse, payload=None, spring=None, home=None, dyn=None,
               cap=None):
    """One controller period in a general environment, per arm:
        F = [B v_xy + S (p - home)_xy, 0] + c + pulse - m g z,   plus the payload's inertia (mass m).
    B, S: (batch, 2, 2) velocity and position gains; c, pulse: (batch, 3) [N]; payload m: (batch,) [kg]."""
    batch = B.shape[0]
    payload = jnp.zeros(batch) if payload is None else payload
    spring = jnp.zeros((batch, 2, 2)) if spring is None else spring
    home = jnp.zeros(3) if home is None else home

    def force(p, v, B, c, f_pulse, mass, S):
        planar = B @ v[:2] + S @ (p - home)[:2]
        weight = jnp.array([0.0, 0.0, -GRAVITY]) * mass
        return jnp.concatenate([planar, jnp.zeros(1)]) + c + f_pulse + weight, jnp.zeros(3)
    return _advance(state, command, noise_scale, noise, force, (B, c, f_pulse, payload, spring), payload, dyn, cap)[0]


def model_terms(d):
    """Handle Jacobian (B, 3, 7) and joint-space mass matrix (B, 7, 7) for model-based controllers."""
    return jax.vmap(handle_jacobian)(d), jax.vmap(lambda d: mjx.full_m(MX, d))(d)
