"""Unit tests for sign_latch.SignLabelLatch (ROS-free).

Scenario names reference the 2026-07-04 stop-sign dropout analysis: the Qwen
label and the geometric board latch must be fused by the board distance AT THE
READ'S CAPTURE TIME, with a readable band [min_dist, max_dist], newest-wins
inside the band, 'none' never erasing, and a per-board lifecycle.
"""
from vlm_planner_py.sign_latch import SignLabelLatch


def make_latch(**kw):
    kw.setdefault('min_dist_m', 3.3)
    kw.setdefault('max_dist_m', 5.0)
    return SignLabelLatch(**kw)


def drive_approach(latch, t0=0.0, d0=8.0, v=0.8, dt=0.1, n=100):
    """Record a constant-speed approach: dist d0 shrinking at v m/s, one frame
    per dt. Returns the (t, dist) list for picking capture times."""
    frames = []
    for i in range(n):
        t = t0 + i * dt
        d = d0 - v * i * dt
        if d <= 0:
            break
        latch.record(t, d)
        frames.append((t, d))
    return frames


def t_at_dist(frames, dist):
    """Capture time of the recorded frame closest to the given distance."""
    return min(frames, key=lambda f: abs(f[1] - dist))[0]


class TestReadBand:
    def test_far_read_rejected(self):
        """A (possibly wrong) read from 6+ m never enters the latch."""
        latch = make_latch()
        frames = drive_approach(latch)
        ok, reason = latch.on_label('left', t_at_dist(frames, 6.5))
        assert not ok and 'too-far' in reason
        assert latch.label is None

    def test_in_band_read_accepted(self):
        latch = make_latch()
        frames = drive_approach(latch)
        ok, reason = latch.on_label('stop', t_at_dist(frames, 4.2))
        assert ok and 'accepted' in reason
        assert latch.label == 'stop'

    def test_below_floor_read_rejected_label_frozen(self):
        """In the crop zone (< min_dist) a confident symbol is likely the NEXT
        board -- it must not overwrite the good earlier read."""
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('stop', t_at_dist(frames, 4.0))[0]
        ok, reason = latch.on_label('left', t_at_dist(frames, 2.0))
        assert not ok and 'frozen-below-floor' in reason
        assert latch.label == 'stop'

    def test_newest_wins_inside_band(self):
        """Recognition refines on approach: a closer read replaces a far one."""
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('winding', t_at_dist(frames, 4.8))[0]
        assert latch.on_label('stop', t_at_dist(frames, 3.6))[0]
        assert latch.label == 'stop'


class TestNoneNeverErases:
    def test_none_in_band_keeps_label(self):
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('stop', t_at_dist(frames, 4.0))[0]
        ok, reason = latch.on_label('none', t_at_dist(frames, 3.6))
        assert not ok and reason == 'not-a-sign-label'
        assert latch.label == 'stop'

    def test_none_below_floor_keeps_label(self):
        """The stop-sign dropout: cropped board -> Qwen says 'none' just before
        the arm window. The latched 'stop' must survive."""
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('stop', t_at_dist(frames, 4.0))[0]
        latch.on_label('none', t_at_dist(frames, 3.0))
        latch.on_label('unparsed', t_at_dist(frames, 2.5))
        assert latch.label == 'stop'


class TestBoardLifecycle:
    def test_clear_board_resets_label(self):
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('stop', t_at_dist(frames, 4.0))[0]
        latch.clear_board()
        assert latch.label is None

    def test_stale_read_of_old_board_rejected_after_clear(self):
        """An in-flight read captured before the old board passed behind must
        not label the NEXT board."""
        latch = make_latch()
        frames = drive_approach(latch)
        t_old = t_at_dist(frames, 4.0)   # captured while board 1 was latched
        latch.clear_board()              # board 1 passed behind
        latch.record(10.5, 4.5)          # board 2 latched, in band
        ok, reason = latch.on_label('left', t_old)
        assert not ok and reason == 'stale-board'
        assert latch.label is None

    def test_next_board_fresh_window(self):
        latch = make_latch()
        drive_approach(latch)
        latch.clear_board()
        latch.record(20.0, 4.4)
        ok, _ = latch.on_label('left', 20.0)
        assert ok and latch.label == 'left'


class TestLabelLock:
    def test_locked_label_rejects_in_band_read(self):
        """After lock() (board within the lock distance), even a confident
        IN-BAND read of a different symbol must not change the decision."""
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('right', t_at_dist(frames, 4.0))[0]
        latch.lock()
        ok, reason = latch.on_label('left', t_at_dist(frames, 3.5))
        assert not ok and 'label-locked' in reason
        assert latch.label == 'right'

    def test_lock_without_label_is_noop(self):
        """Nothing to protect: lock() before any accepted read must not block
        the first legitimate read."""
        latch = make_latch()
        frames = drive_approach(latch)
        latch.lock()
        assert not latch.locked
        assert latch.on_label('stop', t_at_dist(frames, 4.0))[0]

    def test_clear_board_releases_lock(self):
        """The lock is per-board: the next board must be readable again."""
        latch = make_latch()
        frames = drive_approach(latch)
        assert latch.on_label('right', t_at_dist(frames, 4.0))[0]
        latch.lock()
        latch.clear_board()
        assert not latch.locked
        latch.record(30.0, 4.4)
        ok, _ = latch.on_label('left', 30.0)
        assert ok and latch.label == 'left'


class TestAttribution:
    def test_no_board_latched_at_capture_rejected(self):
        """A read from a frame where geometry had no board yet is unattributable."""
        latch = make_latch()
        latch.record(1.0, None)
        ok, reason = latch.on_label('stop', 1.0)
        assert not ok and reason == 'no-board-latched-at-capture'

    def test_no_frames_recorded_rejected(self):
        latch = make_latch()
        ok, reason = latch.on_label('stop', 1.0)
        assert not ok and reason == 'no-frames-recorded'

    def test_stamp_far_from_any_frame_rejected(self):
        latch = make_latch(match_tol_sec=0.5)
        latch.record(5.0, 4.0)
        ok, reason = latch.on_label('stop', 9.0)
        assert not ok and reason == 'no-frame-near-stamp'

    def test_unstamped_read_uses_newest_frame(self):
        """Legacy plain-string labels (no stamp) attribute to the newest frame."""
        latch = make_latch()
        latch.record(1.0, 6.0)
        latch.record(2.0, 4.0)
        ok, reason = latch.on_label('stop', None)
        assert ok and 'accepted at 4.00m' in reason

    def test_history_pruned(self):
        latch = make_latch(history_sec=10.0)
        latch.record(0.0, 4.0)
        latch.record(20.0, None)   # prunes t=0 (older than 20 - 10)
        ok, reason = latch.on_label('stop', 0.0)
        assert not ok and reason == 'no-frame-near-stamp'
