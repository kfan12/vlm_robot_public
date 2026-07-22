"""Maneuver coordination state machine (pure Python, no ROS deps).

Turns the noisy per-frame sign read into an explicit *maneuver* the car is committed
to, with a clean entry (the sign passes behind) and exit (the road is straight again).
See docs/knowledge_base.md §2.19 for the design rationale.

States: 'straight' (default / plain line following), 'left', 'right', 'winding'
(an active turn / winding maneuver), 'stop' (terminal halt intent).

Driven by the planner each tick through two INDEPENDENT inputs:
  - on_sign(pending_label, commit) -- ENTRY. Arm a maneuver while its sign board is
    validated ahead, then commit it on the caller-supplied `commit` edge -- the frame
    the board first comes within a commit distance AHEAD (vlm_node computes this;
    formerly it was the frame the board passed BEHIND). The edge fires once, so a board
    cannot re-trigger the same maneuver frame after frame (it stays within the commit
    distance for several frames, so a naive "label + close ahead" test would re-fire).
  - on_path(pts, step_dist) -- EXIT. Fed ONLY the real lane centerline (never the
    creep/halt paths, which are straight by construction). A maneuver must first be
    ENGAGED (the path bends past engage_tol_rad) before a straight path (< straight_tol_rad)
    can release it -- so the straight APPROACH before the turn cannot end it early -- and
    the straight condition must hold while the robot travels release_dist_m (the distance
    travelled since the last call is passed in as step_dist). A travelled-distance
    threshold, not a frame count, is what survives a WINDING road: a winding road is
    locally straight at each inflection point, but only for a SHORT stretch of travel, so
    requiring a real distance of sustained-straight road (longer than an inflection,
    shorter than a true straightaway) keeps the maneuver from releasing between S-bends.
    It is also speed/rate-independent (a frame count is not).

A sign that passes behind PREEMPTS the current maneuver: maneuvers can chain with no
straight section between them (e.g. winding immediately after a right turn), where the
exit logic would otherwise never release the first maneuver to let the second in.

Limitations:
  - 'stop' is terminal -- cleared only by reset(). The physical halt at the red goal is
    a separate path (see KB §2.16); this state only records the intent.
"""
import math

# State constants.
STRAIGHT = 'straight'
LEFT = 'left'
RIGHT = 'right'
WINDING = 'winding'
STOP = 'stop'

# The states that represent an active turn/winding maneuver (i.e. ones that the
# path-straightness exit logic can release back to STRAIGHT).
DIRECTIONAL = {LEFT, RIGHT, WINDING}


def path_heading_change(pts):
    """Total absolute heading change and arc length of a base_link (x, y) polyline.

    Returns (total_abs_dheading_rad, length_m). The heading change is the sum of
    |Δheading| between consecutive segments, which is rotation-invariant, so the
    base_link frame is fine (no need to transform to odom). A straight path returns a
    near-zero heading change; a 90° corner returns ~pi/2; an S-bend accumulates both
    bends. Fewer than 3 points (no two segments to compare) returns (0.0, length).
    """
    if len(pts) < 2:
        return 0.0, 0.0
    total = 0.0
    length = 0.0
    prev_h = None
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg = math.hypot(dx, dy)
        length += seg
        if seg < 1e-6:  # duplicate point -- no defined heading
            continue
        h = math.atan2(dy, dx)
        if prev_h is not None:
            d = math.atan2(math.sin(h - prev_h), math.cos(h - prev_h))  # wrap to [-pi, pi]
            total += abs(d)
        prev_h = h
    return total, length


class ManeuverStateMachine:
    """Tracks the current maneuver. See module docstring for the entry/exit logic."""

    def __init__(self, engage_tol_rad=0.35, straight_tol_rad=0.20,
                 release_dist_m=2.0, min_len_m=1.0):
        self.engage_tol_rad = engage_tol_rad
        self.straight_tol_rad = straight_tol_rad
        self.release_dist_m = release_dist_m
        self.min_len_m = min_len_m
        self.reset()

    def reset(self):
        self.state = STRAIGHT
        self._armed = None          # label validated ahead, waiting for the pass edge
        self._engaged = False       # has the active maneuver bent past engage_tol yet
        self._straight_dist = 0.0   # metres travelled on continuously-straight road

    def on_sign(self, pending_label, commit):
        """ENTRY. Arm the upcoming maneuver while its board is validated ahead, and
        commit it on the `commit` trigger. `commit` may be asserted EVERY tick while
        the board is inside the commit range (see vlm_node._update_sign_odom): the
        label can arrive AFTER the range is entered (~2 s VLM inference lag), so a
        one-shot edge fired into an unarmed FSM silently loses the whole maneuver
        (loop-3 autotune failure: 'right' latched at 3.49 m vs commit range 3.5 m).
        Returns True when a maneuver was committed THIS call -- the caller uses that
        to stop asserting `commit` for this board, which keeps the once-per-board
        property without racing the label."""
        # Arm whichever directional/stop maneuver is currently validated ahead. 'none'
        # (or any non-maneuver label) leaves the last armed value untouched, so a brief
        # detection dropout just before the commit does not disarm the maneuver.
        if pending_label in DIRECTIONAL or pending_label == STOP:
            self._armed = pending_label

        # Commit on the trigger. The committed sign is ground truth that a NEW
        # maneuver begins now, so it PREEMPTS any unfinished maneuver -- e.g. winding
        # directly after a right turn, where there is no straight section between for the
        # exit logic (on_path) to release the right turn on, so without preemption the
        # winding would never be entered.
        if commit and self._armed is not None:
            if self._armed == STOP:
                self.state = STOP
            else:
                self.state = self._armed
                self._engaged = False
                self._straight_dist = 0.0
            self._armed = None
            return True
        return False

    def on_path(self, pts, step_dist=0.0):
        """EXIT. Release an engaged turn/winding maneuver back to STRAIGHT once the lane
        centerline has been straight for release_dist_m of travel. Call ONLY with the
        real lane centerline (never the creep/halt paths). step_dist is the distance the
        robot has travelled since the previous call (so the debounce is in metres of road,
        not frames -- see class docstring)."""
        if self.state not in DIRECTIONAL:
            return
        change, length = path_heading_change(pts)
        if length < self.min_len_m:  # too short to judge straightness reliably
            return
        if not self._engaged:
            # Wait until the maneuver actually bends before allowing a release, so the
            # straight approach BEFORE the turn cannot end it.
            if change >= self.engage_tol_rad:
                self._engaged = True
            return
        # Engaged: accumulate straight-road travel; a bend resets it. This survives a
        # winding road, which is only briefly straight at each inflection.
        if change <= self.straight_tol_rad:
            self._straight_dist += max(0.0, step_dist)
            if self._straight_dist >= self.release_dist_m:
                self.state = STRAIGHT
                self._engaged = False
                self._straight_dist = 0.0
        else:
            self._straight_dist = 0.0
