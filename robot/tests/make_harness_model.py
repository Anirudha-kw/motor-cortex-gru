"""The harness plant: Menagerie panda_nohand, torque-controlled (actuators removed), 1 ms implicitfast steps, contacts
off, joint damping 1 N m s/rad and the real Panda's joint friction (Gaz et al. 2019).  -> harness_panda.mjb"""
from pathlib import Path

import mujoco

spec = mujoco.MjSpec.from_file(str(Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"))
for a in list(spec.actuators):
    spec.delete(a)
for k in list(spec.keys):
    spec.delete(k)
import os
if float(os.environ.get("TOOL", "0") or 0) > 0:  # a mass on the flange (centre 5 cm past it), e.g. the 0.63 kg on the robot
    tool = spec.body("link7").add_body(name="tool", pos=[0, 0, 0.157])
    tool.mass, tool.inertia, tool.explicitinertial = float(os.environ["TOOL"]), [1e-4] * 3, True
spec.option.timestep = 0.001
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
m = spec.compile()
m.dof_damping[:7] = float(__import__("os").environ.get("DAMPING", "1.0"))
m.dof_frictionloss[:7] = [0.54, 0.87, 0.64, 1.27, 0.84, 0.30, 0.56]
import os
m.dof_frictionloss[:7] *= float(os.environ.get("FRICTION_SCALE", "1"))   # 0: Franka's torque loop cancelling friction
import sys
if os.environ.get("ARMATURE"):                  # test the reflected motor inertia (Menagerie: 0.1 on every joint)
    m.dof_armature[:7] = [float(v) for v in os.environ["ARMATURE"].split(",")]
out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("harness_panda.mjb"))
mujoco.mj_saveModel(m, out, None)
print("armature", list(m.dof_armature[:7]), "->", out)
