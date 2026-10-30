# Running the controller on a Franka Panda

Run these on the robot PC, in order, with the e-stop in hand from step 5 on. Stop at the first step that doesn't do
what's described. Placeholders: `<user>@<robot-pc>` is the robot PC's login, `<robot-ip>` the arm's FCI address.

Requirements: ROS Noetic, `franka_ros` and `libfranka`, and a catkin package named `panda` that already builds a
controller library (see "Install" in [README.md](README.md); adapt `install_robot.sh`'s two anchors to your package).

## 0. Before anything moves

- **Clear the workspace around the start posture.** `q_home = [0, -0.35, 0, -2.3, 0, 1.95, 0.785]` puts the hand
  (10 cm past the flange, which points down) at about (0.44, 0, 0.39) m in the base frame. Reaches stay within 6 cm
  of it horizontally. Remove or account for any tool that could reach the table.
- **Payload.** A light handle at the flange is fine: training covered up to 0.6 kg of unmodelled tool. Declare a
  mounted tool's mass in the robot's load settings as you normally do.
- **Collision thresholds.** Reaches peak at about 0.4 m/s with up to about 16 N·m of commanded torque. On our arm, a
  10–12 Hz joint-2 limit cycle made Franka's estimated external force swing by about ±50 N, and its Cartesian
  reflex (40 N) occasionally fired mid-episode. If that happens on normal reaches, raise the Cartesian force
  thresholds in `franka_control`'s config. Don't loosen the controller's own trips.
- **Damping.** The real arm delivers torque later and weaker than the training plants, and the undamped policy
  oscillates vertically until a trip. Our sessions used `--damping 20` (N·m·s/rad on joints 1–4, ×0.2 on 5–6, ×0.1
  on 7). The controller's own default is 0 and the experiment script's is 12, so pass it explicitly.

## 1. Copy the package (from the workstation, in the repo root)

```bash
rsync -av --exclude tests robot/ <user>@<robot-pc>:~/motor_cortex_bundle/
```

## 2. Isolated build (your package is untouched)

```bash
bash ~/motor_cortex_bundle/install_robot.sh
```

It checks the bundle hashes and the two anchors in the `panda` package, copies the package to `/tmp`, adds the
controller there and builds it (`FRANKA_DIR`, default `~/libfranka/build`). Expect `isolated build OK`.

## 3. Install (controller launch stopped)

With the arm stationary and the Franka launch stopped:

```bash
bash ~/motor_cortex_bundle/install_robot.sh install
```

It backs up `~/catkin_ws/src/panda` and `libpanda_controllers.so` to `~/motor_cortex_backup.*` and prints the path.
To revert, copy the backup back and rebuild.

## 4. Launch (the arm holds where it is)

Start `roscore` in its own terminal first if you run more than one launch file. Then:

```bash
source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
roslaunch panda motor_cortex_controller.launch robot_ip:=<robot-ip> load_gripper:=false
```

`first_session:=true` is the default: torque caps 20/5 N·m and 6 cm reaches. In a second terminal:

```bash
rostopic echo -n 1 /motor_cortex_controller/status      # 14 = hand-at-home differs from the training model by > 3 mm
rostopic echo -n 1 /motor_cortex_controller/telemetry   # [9:12] = hand position: should be plausible
```

## 5. One damped episode, no field

```bash
rosrun panda motor_cortex_experiment.py --blocks null:1 --n-out 6 --damping 20 --record ~/mc_null.npz
```

The script homes the arm (8 s, status 1 then 0), sets the damping (status 15) and waits for Enter. Expect six 6 cm
reaches out and back (8.4 s), then status 10 (episode done) and `session complete`.

## 6. Sessions

The script homes before every episode, applies the curl field (35.4 N·s/m) only during curl episodes, and stops at
the first trip or refusal. It saves its recording even after a trip or Ctrl-C. `--yes` skips the Enter before each
episode.

- **Balanced session** (adaptation within an episode). Pairs of null and curl episodes; each pair shares a new target
  order, and which comes first alternates:
  ```bash
  rosrun panda motor_cortex_experiment.py --pairs 20 --seed 100 --n-out 6 --damping 20 --yes --record ~/mc_balanced.npz
  python3 tests/analyze_balanced.py mc_balanced.npz       # on the workstation
  ```
- **Catch-trial session** (is the adaptation feedforward?). As above, but the field switches off as the sixth outward
  reach starts:
  ```bash
  rosrun panda motor_cortex_experiment.py --pairs 20 --seed 200 --off-after 5 --n-out 6 --damping 20 --yes --record ~/mc_catch.npz
  python3 tests/catch_score.py mc_catch.npz                # several files if the session was resumed
  ```

The hypotheses, predictions and outcomes for each session are in [tests/PREDICTIONS.md](tests/PREDICTIONS.md),
written before each run.

## Trips and reflexes

Status 21–25 means the controller switched to damping only: 21 workspace, 22 joint speed, 23 joint limit,
24 non-finite output, 25 hand speed. Inspect the arm, then:

```bash
rostopic pub -1 /motor_cortex_controller/command std_msgs/String "reset"    # back to hold where it is
```

A Franka reflex (`libfranka: Move command aborted: motion aborted by reflex!`) stops the arm outside the controller.
The controller's episode clock runs on, so the script sees the next episode refused (`not at home`) and stops.
Relaunch (step 4): a fresh launch holds the arm where it is.

## Commands and logs

| Command (`/motor_cortex_controller/command`, `std_msgs/String`) | Effect |
|---|---|
| `home` | 8 s minimum-jerk move to `q_home` (refused from more than 1 rad away) |
| `start [n_out] [seed]` | an episode of `n_out` outward reaches, each followed by a return (only from home, arm still) |
| `stop` | back to hold where the arm is |
| `reset` | leave TRIPPED, back to hold |
| `damping <k>` | joint damping `k` (0–30), in hold only, from the next start |
| `pulse <joint> <N m> <ms>` | a torque pulse on one joint in hold, \|N m\| ≤ 5, 10–200 ms (system identification) |

The field is set on `/motor_cortex_controller/field`. Three logs are published: `telemetry` (100 Hz), `policy_log`
(every policy tick: the exact 28 inputs, the output and the torque) and `fast_log` (1 kHz: q, dq, measured `tau_J`,
`tau_J_d`, the torque sent, Franka's gravity and `tau_ext_hat_filtered`). `motor_cortex_experiment.py --record`
saves all three per episode, and `--pulses "2,4,150;2,-4,150"` runs pulses in hold.
