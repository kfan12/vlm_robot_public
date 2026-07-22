"""Distance-gated latch for VLM sign labels (ROS-free, unit-tested).

Problem (KB: stop-sign dropout, 2026-07-04): the Qwen label on /vlm/sign is
frame-global and sticky, while the geometric board latch (vlm_node.sign_odom)
is per-board. Fusing "latest label" with "current board distance" fails two
ways:

  * A read taken FAR away (unreliable — small board, past misreads) sticks
    around and gets armed once the board later comes within reach.
  * A read taken too CLOSE returns 'none' (the pitched-down camera crops the
    board top from ~3.3 m board distance, the STOP letters from ~2.7 m) or —
    with a second sign in view — the NEXT board's symbol, and overwrites a
    perfectly good earlier read right when the arbitration needs it.

Fix: attribute every read to the board that was latched WHEN ITS FRAME WAS
CAPTURED (inference takes ~2 s — the board has moved ~1.6 m closer by the time
the label arrives), and accept it only if that board sat inside a readable
distance band:

    dist > max_dist_m   -> reject (far reads are unreliable)
    dist < min_dist_m   -> reject (board crops out of frame; a confident label
                           here is likely the NEXT sign — the latched label is
                           FROZEN below the floor)
    in band             -> accept; the NEWEST accepted read wins, so the
                           recognition refines as the board grows on approach
    'none'/'unparsed'   -> NEVER erase: with the board still geometrically
                           latched ahead, "can't read it" is occlusion/crop,
                           not absence. Existence is the geometry's call;
                           meaning is the VLM's.

The caller records (frame_stamp, board_dist) every planning tick and bumps the
board identity when the latched board passes behind; a label whose capture
frame belongs to a previous board is rejected as stale.
"""
from collections import deque

SIGN_LABELS = ('left', 'right', 'winding', 'stop')


class SignLabelLatch:
    def __init__(self, min_dist_m=3.3, max_dist_m=5.0,
                 history_sec=10.0, match_tol_sec=0.5):
        """min/max_dist_m: readable band on the latched-board distance at the
        label's CAPTURE time. history_sec: how much (stamp, dist) history to
        keep for capture-time lookup. match_tol_sec: max |stamp difference|
        between the label's capture stamp and the nearest recorded frame."""
        self.min_dist_m = float(min_dist_m)
        self.max_dist_m = float(max_dist_m)
        self.history_sec = float(history_sec)
        self.match_tol_sec = float(match_tol_sec)
        self._hist = deque()      # (t, dist_or_None, board_id)
        self._board_id = 0
        self.label = None         # accepted label for the CURRENT board
        self.locked = False       # label immutable until the board passes (lock())

    def record(self, t, dist):
        """Record one planning frame: capture stamp t [s] and the latched board
        distance at that frame (None = no board latched). Call every tick,
        AFTER the board latch update for the frame."""
        self._hist.append((float(t), dist, self._board_id))
        while self._hist and self._hist[0][0] < t - self.history_sec:
            self._hist.popleft()

    def lock(self):
        """Freeze the current label until the board passes behind. Called by
        the node once the tracked board comes within the lock distance
        (~3 m): from here to the commit point the board escapes the camera
        view, so no further read — of this board's clipped remains or of the
        NEXT board entering the frame — may change the decision. No-op
        without a latched label (nothing to protect)."""
        if self.label is not None:
            self.locked = True

    def clear_board(self):
        """The latched board passed behind: next board gets a fresh identity
        and an empty label. Frames already recorded keep the OLD identity, so
        an in-flight read of the old board cannot leak onto the new one."""
        self._board_id += 1
        self.label = None
        self.locked = False

    def on_label(self, label, t_capture=None):
        """Offer a VLM read. t_capture = the label's image stamp [s]; None
        falls back to the newest recorded frame (legacy, unstamped labels).
        Returns (accepted: bool, reason: str) for logging."""
        if label not in SIGN_LABELS:
            # 'none'/'unparsed' never erase a latched label (see module doc).
            return False, 'not-a-sign-label'
        if self.locked:
            return False, f'label-locked ({self.label!r} kept)'
        if not self._hist:
            return False, 'no-frames-recorded'
        if t_capture is None:
            t, dist, board_id = self._hist[-1]
        else:
            t, dist, board_id = min(self._hist,
                                    key=lambda e: abs(e[0] - t_capture))
            if abs(t - t_capture) > self.match_tol_sec:
                return False, 'no-frame-near-stamp'
        if board_id != self._board_id:
            return False, 'stale-board'
        if dist is None:
            return False, 'no-board-latched-at-capture'
        if dist > self.max_dist_m:
            return False, f'too-far ({dist:.2f}m > {self.max_dist_m:.2f}m)'
        if dist < self.min_dist_m:
            return False, f'frozen-below-floor ({dist:.2f}m < {self.min_dist_m:.2f}m)'
        self.label = label
        return True, f'accepted at {dist:.2f}m'
