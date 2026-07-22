# Troubleshooting

Known issues and their fixes, roughly in the order you'd hit them.

## Nothing opens / the demo window flashes and disappears

- **tmux missing** — `sudo apt install -y tmux`.
- **Workspace not built** — the launcher warns if `install/setup.bash` is
  missing. Run `colcon build --symlink-install` in the repo root first.
- **WSL2: no `wt.exe`** — the launcher opens the demo in a new window via
  Windows Terminal. If neither `wt.exe` nor `wezterm.exe` is on the PATH it
  prints the manual command instead: `tmux attach -t vlm_robot_demo`.
  (Windows Terminal is preinstalled on Windows 11; on older setups install it
  from the Microsoft Store.)
- A pane that errors **stays open** with the message visible — read it there.

## Gazebo crashes at startup (Ogre / rendering errors)

`export LIBGL_ALWAYS_SOFTWARE=1` **must** be set before `ign gazebo` runs on
WSL2 (the GPU GL path crashes in Ogre's texture copy). Add it to `~/.bashrc`.
Native Ubuntu with a working GPU driver doesn't need it, but it's harmless.

## Gazebo is up but the camera image freezes (car drives blind / stops)

WSL2 rendering lottery: on some boots the sensor camera freezes after
~40–90 s (bit-identical frames). This is per-boot luck — kill the demo
(`tmux kill-session -t vlm_robot_demo`) and relaunch. Related: **never** run
`ign gazebo --headless-rendering` on WSL — that flag freezes the camera
reliably; the demo doesn't use it.

## The car doesn't move

- Wait the full startup stagger (~40 s) — the planner is deliberately last.
- Check pane 3 (MPC) and pane 6 (planner) for errors; check
  `ros2 topic hz /vlm_path_odom` and `/cmd_vel` in pane 8.
- `MANUAL_DRIVE=1` disables the controller by design — drive in pane 8.

## `colcon build` fails on `mpc_tracker_cpp` (OsqpEigen not found)

OSQP + osqp-eigen must be source-installed into `/usr/local`
(`scripts/setup/install_deps.sh` does this; manual steps in
[INSTALL.md §2.4](INSTALL.md)). Verify
`/usr/local/lib/cmake/OsqpEigen/OsqpEigenConfig.cmake` exists, then rebuild.
Note colcon does not search `/usr/local` by itself — the package's
CMakeLists already sets `OsqpEigen_DIR` accordingly; don't remove that line.

## Signs look almost black in the camera image (WSL)

The software-GL renderer darkens the sign textures. The detection code
already compensates (HSV value floor), so this is cosmetic — but if you
retune color thresholds, tune against the *rendered* frames (pane 5 saves
them to `captures/`), not the source PNGs.

## VLM extra (`VLM_SIGN=1`) issues

- **`torch not found` in pane 7** — the venv is missing; run
  `./scripts/setup/setup_vlm_venv.sh`. The sign node must run with the venv
  python via `python3 -m` (the demo script handles this).
- **CUDA out of memory** — Qwen2.5-VL-3B 4-bit needs ~6 GB VRAM with nothing
  else on the GPU. Close other GPU apps.
- **cv_bridge `_ARRAY_API not found` warnings** — expected under the venv's
  NumPy 2; the VLM nodes don't use cv_bridge, so it's noise. Do not
  "fix" it by downgrading numpy.
- **First run is slow** — the model (~3 GB) downloads to
  `~/.cache/huggingface` once.

## Gazebo slow / topics missing on WSL

Try `export IGN_IP=127.0.0.1` (multicast discovery can stall under WSL's
virtual network).

## tmux crash course

- Switch pane: `Ctrl-b` then arrow key. Scroll: `Ctrl-b` `[`, arrows/PgUp,
  `q` to exit scroll mode.
- Kill everything: `tmux kill-session -t vlm_robot_demo` (or just close the
  window — the session dies with it).
- Re-attach from any terminal: `tmux attach -t vlm_robot_demo`.
