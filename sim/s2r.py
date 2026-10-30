"""Sim-to-real randomisation for the in-context GRU on the Panda (meta.py), before it drives the real arm.

What the simulated plant lacks and the real Panda has, drawn per episode and per arm (panda.Dyn):
  friction    Coulomb friction up to FRICTION_MAX per joint (Gaz et al. 2019 identified about 0.3-1.3 N m on the
              real Panda), smoothed as f tanh(qd / 0.05 rad/s)
  viscous     joint damping x VISCOUS around the model's 1 N m s/rad
  bias        a constant torque offset per joint up to BIAS_MAX: gravity compensation that is slightly wrong
  gain        torque gain error per joint (torque sensing and motor constants)
  mass_scale  each link's mass and inertia x MASS
  tool        a known, gravity-compensated tool at the handle (inertia only) up to TOOL_MAX kg
  lag         the torque loop's time constant in LAG
  delay       NOT randomised (P_DELAY2 = 0): a second step of delay made the gradient through the 840-step closed
              loop explode (norm 1e9-1e11 against 0.04; s2r_diag.py), and the delay is ours to set -- the robot senses
              in 1-2 ms, so the deployment loop feeds the policy the state of exactly one period ago (10 ms)
  noise       the sensor noise x NOISE
  armature    (with armature=True) each joint's reflected motor inertia, log-uniform in ARMATURE.  The Menagerie
              model's 0.1 kg m^2 on every joint was the gap that made the first hardware reach oscillate vertically
              (shoulder and elbow saturating at the torque caps): simulation with 0.01 reproduced it, 0.1 did not
  real        (real=dict(...), meta_train --lag-joint/--gain/--ctrl-damping/--friction-scale) what the logged
              hardware reaches showed (2026-10-20, robot/tests/plant_fit2.py): the real arm
              delivers torque late and weak -- per-joint effective lag 25-60 ms on the shoulder and elbow, gain
              0.45-0.85 -- far outside LAG and GAIN above.  The lag becomes per joint (log-uniform), the gain range
              widens, and the robot controller's joint damping D = k DAMPING_SHAPE (k uniform per arm) is part of the
              loop, so one policy works across the damping the robot may need
and, always, the deployment controller's limits: commands capped at CAP and rate-limited to 1000 N m/s (Franka's
limit), as the real-time loop on the robot will enforce them.  A curriculum scales every range by `scale` (0 =
nominal values, 1 = full ranges).
"""
import jax
import jax.numpy as jnp

from panda import N_JOINTS, TAU_MOTOR, Dyn

FRICTION_MAX = jnp.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])     # [N m]
BIAS_MAX = jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])         # [N m]
VISCOUS = (0.5, 2.0)                                              # x the model's 1 N m s/rad
GAIN = (0.9, 1.1)
MASS = (0.85, 1.15)
TOOL_MAX = 0.6                                                    # [kg]
LAG = (0.005, 0.02)                                               # [s]
NOISE = (0.5, 2.0)
P_DELAY2 = 0.0                                                    # see the header
CAP = jnp.array([40.0, 40.0, 40.0, 40.0, 8.0, 8.0, 8.0])         # safety caps on the commanded torque [N m]
ARMATURE_NOMINAL = 0.1                                            # the training model's value [kg m^2]
ARMATURE = (0.005, 0.1)                                           # log-uniform range [kg m^2] (armature=True)
DAMPING_SHAPE = jnp.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.1])  # the robot controller's "damping k" per joint
# Plants fitted jointly to the undamped (tripped) and damping-12 real reaches (plant_fit2.py, top 3; tool 0.63 kg,
# no joint damping).  Probes only: training samples the ranges.
REAL_FITS = [
    dict(armature=0.083, gain=[0.75, 0.84, 0.93, 0.86, 0.94, 0.55, 0.99], lag=[.011, .045, .022, .024, .051, .003, .039],
         friction=[0.6, 1.6, 0.4, 0.9, 0.1, 0.8, 0.2]),
    dict(armature=0.023, gain=[0.43, 0.91, 0.59, 0.52, 0.83, 0.69, 0.61], lag=[.003, .058, .015, .009, .015, .009, .029],
         friction=[1.7, 0.6, 1.9, 0.6, 0.4, 0.9, 0.4]),
    dict(armature=0.025, gain=[0.93, 0.94, 0.83, 0.64, 0.55, 0.50, 0.57], lag=[.004, .051, .044, .018, .013, .011, .034],
         friction=[1.8, 0.7, 0.5, 1.9, 0.4, 1.0, 0.0]),
]


def _between(key, shape, lo, hi, nominal, scale):
    """Uniform in [nominal + scale (lo - nominal), nominal + scale (hi - nominal)]."""
    lo, hi = nominal + scale * (lo - nominal), nominal + scale * (hi - nominal)
    return jax.random.uniform(key, shape, minval=lo, maxval=hi)


def sample_dyn(key, batch, scale=1.0, drop=(), armature=None, real=None):
    """drop: field names left at their nominal values (e.g. ('bias', 'viscous'), whose gradient tails were the
    heaviest in s2r_diag.py).  armature: None (the model's 0.1) or a (lo, hi) range sampled log-uniformly per joint,
    the range growing from 0.1 with `scale`.  real: optional dict with lag (lo, hi) [s] per joint log-uniform,
    gain (lo, hi), friction_scale (x FRICTION_MAX), ctrl_damping (lo, hi) for k; lag/gain/friction grow with
    `scale`, the damping (a controller setting, not a plant error) does not."""
    full = _sample(key, batch, scale)
    if real:
        k = jax.random.split(jax.random.fold_in(key, 7), 4)
        j = (batch, N_JOINTS)
        if "lag" in real:
            lo, hi = (jnp.log(TAU_MOTOR) + scale * (jnp.log(v) - jnp.log(TAU_MOTOR)) for v in real["lag"])
            full = full._replace(lag=jnp.exp(lo + jax.random.uniform(k[0], j) * (hi - lo)))
        if "gain" in real:
            full = full._replace(gain=_between(k[1], j, *real["gain"], 1.0, scale))
        if "friction_scale" in real:
            full = full._replace(friction=full.friction * real["friction_scale"])
        if "ctrl_damping" in real:
            kd = jax.random.uniform(k[3], (batch, 1), minval=real["ctrl_damping"][0], maxval=real["ctrl_damping"][1])
            full = full._replace(ctrl_damping=kd * DAMPING_SHAPE)
        if "viscous" in real:                     # absolute joint damping range [N m s/rad]; the fits have none
            full = full._replace(viscous=_between(jax.random.fold_in(k[1], 3), j, *real["viscous"], 1.0, scale))
        if real.get("fit_mix", 0.0) > 0:          # a share of arms drawn around the plants fitted to the real arm
            full = _mix_fits(jax.random.fold_in(key, 11), full, real["fit_mix"])
    full = full._replace(**{f: getattr(nominal_dyn(batch), f) for f in drop})
    if armature is None:
        return full
    lo, hi = (jnp.log(ARMATURE_NOMINAL) + scale * (jnp.log(v) - jnp.log(ARMATURE_NOMINAL)) for v in armature)
    u = jax.random.uniform(jax.random.fold_in(key, 99), (batch, N_JOINTS))
    return full._replace(armature=jnp.exp(lo + u * (hi - lo)))


def _sample(key, batch, scale):
    k = jax.random.split(key, 9)
    j = (batch, N_JOINTS)
    return Dyn(friction=scale * FRICTION_MAX * jax.random.uniform(k[0], j),
               viscous=_between(k[1], j, *VISCOUS, 1.0, scale),
               bias=scale * BIAS_MAX * jax.random.uniform(k[2], j, minval=-1.0, maxval=1.0),
               gain=_between(k[3], j, *GAIN, 1.0, scale),
               mass_scale=_between(k[4], j, *MASS, 1.0, scale),
               tool=scale * TOOL_MAX * jax.random.uniform(k[5], (batch,)),
               lag=_between(k[6], (batch,), *LAG, TAU_MOTOR, scale),
               delay=jnp.where(jax.random.uniform(k[7], (batch,)) < scale * P_DELAY2, 2, 1),
               noise=_between(k[8], (batch,), *NOISE, 1.0, scale))


def nominal_dyn(batch):
    """The nominal values (with the deployment limits still applied by the rollout's cap)."""
    j = (batch, N_JOINTS)
    return Dyn(friction=jnp.zeros(j), viscous=jnp.ones(j), bias=jnp.zeros(j), gain=jnp.ones(j),
               mass_scale=jnp.ones(j), tool=jnp.zeros(batch), lag=jnp.full(batch, TAU_MOTOR),
               delay=jnp.ones(batch, int), noise=jnp.ones(batch))


def _mix_fits(key, dyn, share):
    """Replace a `share` of the arms by jittered copies of REAL_FITS: gain x U(0.85, 1.15), lag x U(0.7, 1.3),
    friction x U(0.5, 1.5), armature x U(0.7, 1.3), viscous U(0, 0.5), the 0.63 kg tool; the controller damping
    and everything else of the arm are kept."""
    B = dyn.gain.shape[0]
    k = jax.random.split(key, 7)
    j = (B, N_JOINTS)
    fits = {f: jnp.array([REAL_FITS[i][f] for i in range(len(REAL_FITS))]) for f in ("gain", "lag", "friction")}
    arm = jnp.array([REAL_FITS[i]["armature"] for i in range(len(REAL_FITS))])
    which = jax.random.randint(k[0], (B,), 0, len(REAL_FITS))
    u = lambda kk, lo, hi: jax.random.uniform(kk, j, minval=lo, maxval=hi)
    gain = fits["gain"][which] * u(k[1], 0.85, 1.15)
    lag = jnp.maximum(fits["lag"][which] * u(k[2], 0.7, 1.3), 0.002)
    fric = fits["friction"][which] * u(k[3], 0.5, 1.5)
    armature = arm[which][:, None] * u(k[4], 0.7, 1.3)
    on = jax.random.uniform(k[5], (B,)) < share
    lag_all = jnp.broadcast_to(dyn.lag if dyn.lag.ndim == 2 else dyn.lag[:, None], j)
    arm_all = dyn.armature if dyn.armature is not None else jnp.full(j, ARMATURE_NOMINAL)
    pick = lambda a, b: jnp.where(on[:, None], a, b)
    return dyn._replace(gain=pick(gain, dyn.gain), lag=pick(lag, lag_all), friction=pick(fric, dyn.friction),
                        armature=pick(armature, arm_all), viscous=pick(u(k[6], 0.0, 0.5), dyn.viscous),
                        tool=jnp.where(on, 0.63, dyn.tool))


def real_fit_dyn(batch, fit=0, damping=0.0):
    """A plant fitted to the logged hardware reaches (REAL_FITS), with the robot controller's damping k."""
    f = REAL_FITS[fit]
    j = (batch, N_JOINTS)
    row = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float32), j)
    return Dyn(friction=row(f["friction"]), viscous=jnp.zeros(j), bias=jnp.zeros(j), gain=row(f["gain"]),
               mass_scale=jnp.ones(j), tool=jnp.full(batch, 0.63), lag=row(f["lag"]), delay=jnp.ones(batch, int),
               noise=jnp.ones(batch), armature=jnp.full(j, f["armature"]),
               ctrl_damping=jnp.broadcast_to(damping * DAMPING_SHAPE, j))


def hard_dyn(batch):
    """A fixed worst case for monitoring: every range at its harder end."""
    j = (batch, N_JOINTS)
    return Dyn(friction=jnp.broadcast_to(FRICTION_MAX, j), viscous=jnp.full(j, VISCOUS[1]),
               bias=jnp.broadcast_to(BIAS_MAX, j), gain=jnp.full(j, GAIN[0]), mass_scale=jnp.full(j, MASS[1]),
               tool=jnp.full(batch, TOOL_MAX), lag=jnp.full(batch, LAG[1]), delay=jnp.ones(batch, int),
               noise=jnp.full(batch, NOISE[1]))
