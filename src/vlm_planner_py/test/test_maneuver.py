"""Script-style checks for the maneuver state machine (pure logic, no ROS/Gazebo).

Run:  python3 src/vlm_planner_py/test/test_maneuver.py
Mirrors test_opencv.py's style: prints each scenario's result and asserts the expected
state transitions. Exits non-zero if any assertion fails.
"""
import math
import sys
sys.path.insert(0, 'src/vlm_planner_py')
from vlm_planner_py.maneuver import (  # noqa: E402
    ManeuverStateMachine, path_heading_change, STRAIGHT, LEFT, RIGHT, WINDING, STOP)


def straight_path(n=12, spacing=0.3):
    """Straight base_link path along +x."""
    return [(i * spacing, 0.0) for i in range(n)]


def arc_path(total_turn_rad, n=12, spacing=0.3):
    """Constant-curvature arc accumulating total_turn_rad of heading over n segments."""
    dpsi = total_turn_rad / (n - 1)
    pts = [(0.0, 0.0)]
    x, y, psi = 0.0, 0.0, 0.0
    for _ in range(n - 1):
        psi += dpsi
        x += spacing * math.cos(psi)
        y += spacing * math.sin(psi)
        pts.append((x, y))
    return pts


# Per-call robot travel fed to on_path; with release_dist_m=1.5 below, 3 straight calls
# (3 * 0.6 = 1.8 >= 1.5) release on the 3rd -- mirrors the old 3-frame debounce.
STEP = 0.6


def sm():
    return ManeuverStateMachine(engage_tol_rad=0.35, straight_tol_rad=0.20,
                                release_dist_m=1.5, min_len_m=1.0)


def check(name, got, want):
    ok = got == want
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: state={got!r} (want {want!r})')
    assert ok, f'{name}: got {got!r}, want {want!r}'


# --- path_heading_change sanity --------------------------------------------------
ch, ln = path_heading_change(straight_path())
print(f'straight path: heading_change={ch:.3f} rad, length={ln:.2f} m')
assert ch < 0.05 and ln > 1.0
ch, ln = path_heading_change(arc_path(math.pi / 2))  # 90 deg corner
print(f'90deg arc:     heading_change={ch:.3f} rad, length={ln:.2f} m')
# Summed segment-to-segment change of a discretized arc is (n-2)/(n-1)*total, slightly
# under the continuous pi/2 -- well above engage_tol either way.
assert 1.2 < ch < math.pi / 2

# --- Scenario 1: left turn full lifecycle ----------------------------------------
print('\nScenario 1: arm left, no early release on straight approach, '
      'enter on pass, engage, release.')
m = sm()
# Board validated ahead but not passed: arm only, stay straight.
m.on_sign('left', commit=False)
m.on_path(straight_path(), STEP)          # straight approach must NOT release (not entered)
check('armed, approach straight', m.state, STRAIGHT)
# Board passes behind -> commit the maneuver.
m.on_sign('none', commit=True)
check('left committed on pass', m.state, LEFT)
# Still straight for one frame before the curve -> engaged? no, so no release.
m.on_path(straight_path(), STEP)
check('post-entry straight (not engaged)', m.state, LEFT)
# Now the path bends through the corner -> engage (no release while bent).
m.on_path(arc_path(math.pi / 2))
check('engaged on bend', m.state, LEFT)
# Road straight again: needs release_frames consecutive straight frames.
m.on_path(straight_path(), STEP)
check('straight frame 1', m.state, LEFT)
m.on_path(straight_path(), STEP)
check('straight frame 2', m.state, LEFT)
m.on_path(straight_path(), STEP)
check('straight frame 3 -> release', m.state, STRAIGHT)

# --- Scenario 2: winding does not release on a single straight frame -------------
print('\nScenario 2: winding, single straight frame between S-bends must not release.')
m = sm()
m.on_sign('winding', commit=False)
m.on_sign('none', commit=True)
check('winding committed', m.state, WINDING)
m.on_path(arc_path(0.8))            # first bend -> engage
check('winding engaged', m.state, WINDING)
m.on_path(straight_path(), STEP)          # lone straight frame between bends (count=1)
check('one straight between bends', m.state, WINDING)
m.on_path(arc_path(-0.8))           # opposite bend resets the debounce counter
check('opposite bend', m.state, WINDING)
m.on_path(straight_path(), STEP)
m.on_path(straight_path(), STEP)
m.on_path(straight_path(), STEP)          # now 3 in a row -> release
check('winding release after 3 straight', m.state, STRAIGHT)

# --- Scenario 3: stop is terminal until reset ------------------------------------
print('\nScenario 3: stop commits and is terminal; reset clears it.')
m = sm()
m.on_sign('stop', commit=False)
m.on_sign('none', commit=True)
check('stop committed', m.state, STOP)
m.on_path(straight_path(), STEP)          # path logic never releases stop
check('stop survives straight path', m.state, STOP)
m.reset()
check('reset -> straight', m.state, STRAIGHT)

# --- Scenario 4: no re-entry mid-maneuver; pass with nothing armed is a no-op -----
print('\nScenario 4: pass edge with nothing armed does nothing.')
m = sm()
m.on_sign('none', commit=True)
check('pass, nothing armed', m.state, STRAIGHT)

# --- Scenario 5: too-short path cannot engage (min_len guard) --------------------
print('\nScenario 5: a too-short bent path cannot engage the release logic.')
m = sm()
m.on_sign('left', commit=False)
m.on_sign('none', commit=True)
short_arc = arc_path(math.pi / 2, n=3, spacing=0.2)  # length ~0.4 m < min_len 1.0
m.on_path(short_arc)
# Not engaged (too short), so even a long straight path afterwards cannot release.
m.on_path(straight_path(), STEP)
m.on_path(straight_path(), STEP)
m.on_path(straight_path(), STEP)
check('short bend did not engage -> stays left', m.state, LEFT)

# --- Scenario 6: maneuvers chain -- winding preempts an unreleased right turn --------
print('\nScenario 6: winding sign passing behind preempts a still-active right turn '
      '(no straight section between).')
m = sm()
m.on_sign('right', commit=False)
m.on_sign('none', commit=True)
check('right committed', m.state, RIGHT)
m.on_path(arc_path(math.pi / 2))    # in the right turn, engaged, never straightens...
check('right engaged, no straight between', m.state, RIGHT)
# Winding sign comes up and passes behind while still mid-turn -> preempt to winding.
m.on_sign('winding', commit=False)
m.on_sign('none', commit=True)
check('winding preempts right', m.state, WINDING)
# And the winding can still release normally once truly straight.
m.on_path(arc_path(0.8))
for _ in range(3):
    m.on_path(straight_path(), STEP)
check('winding releases after chain', m.state, STRAIGHT)

# --- Scenario 7: winding inflection -- a sub-threshold straight RUN must not release --
print('\nScenario 7: a winding inflection (straight for < release_dist_m of travel) '
      'does not release.')
m = sm()
m.on_sign('winding', commit=False)
m.on_sign('none', commit=True)
m.on_path(arc_path(0.8))            # engage on the first bend
# Inflection: straight, but only 2 frames (2*0.6 = 1.2 m < 1.5 m release_dist).
m.on_path(straight_path(), STEP)
m.on_path(straight_path(), STEP)
check('2 straight frames at inflection (1.2 m < 1.5 m)', m.state, WINDING)
m.on_path(arc_path(-0.8))           # next bend -> accumulator resets
check('next bend resets accumulator', m.state, WINDING)
# True straightaway now: travel past release_dist_m -> release.
for _ in range(3):
    m.on_path(straight_path(), STEP)
check('true straightaway releases', m.state, STRAIGHT)

print('\nAll maneuver state-machine checks passed.')
