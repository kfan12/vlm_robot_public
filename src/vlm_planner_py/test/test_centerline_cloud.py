"""Unit tests for centerline_from_line_cloud's boundary-side classification and
anchor — the pieces whose failure modes caused the turn oscillation:

* a lone boundary ridden at y~0 must be sided by MEMORY, not by its noise sign;
* x-disjoint fragments of ONE boundary must never be forced onto opposite sides;
* the near anchor must not import the lane's own curvature as a phantom offset;
* fold-back (sawtooth) centerlines must be detectable so the node can reject
  the frame instead of publishing it.

All inputs are synthetic clouds in base_link (x fwd, y left), ROS-free.
"""
import math

from vlm_planner_py.centerline import centerline_from_line_cloud, path_folds_back

HALF_LANE = 0.5


def _line(x0, x1, y_fn, step=0.05):
    """Dense boundary points along x with lateral y_fn(x)."""
    n = int((x1 - x0) / step) + 1
    return [(x0 + i * step, y_fn(x0 + i * step)) for i in range(n)]


def _mid_lateral(pts, x_query):
    for i in range(1, len(pts)):
        (xa, ya), (xb, yb) = pts[i - 1], pts[i]
        if xa <= x_query <= xb and xb > xa:
            return ya + (x_query - xa) / (xb - xa) * (yb - ya)
    return None


def test_two_straight_boundaries_centered():
    """Robot centered between two straight lines -> centerline ~ y=0, anchor ~0."""
    cloud = _line(0.8, 3.0, lambda x: 0.5) + _line(0.8, 3.0, lambda x: -0.5)
    out = centerline_from_line_cloud(cloud, half_lane=HALF_LANE)
    assert out[0][0] == 0.0
    assert abs(out[0][1]) < 0.05
    for (x, y) in out:
        assert abs(y) < 0.1


def test_lone_boundary_sided_by_memory_not_noise_sign():
    """Riding ON the right line: its lateral sign is noise (+/-2 cm). With memory
    seeded by a two-boundary frame, both noise signs must yield the SAME
    centerline side (to the LEFT of the line), never a full-lane flip."""
    mem = {}
    seed = _line(0.8, 3.0, lambda x: 0.55) + _line(0.8, 3.0, lambda x: -0.45)
    centerline_from_line_cloud(seed, half_lane=HALF_LANE, side_memory=mem)
    for noise in (+0.02, -0.02):
        lone = _line(0.8, 3.0, lambda x: noise)
        out = centerline_from_line_cloud(lone, half_lane=HALF_LANE,
                                         side_memory=dict(mem))
        mid = _mid_lateral(out, 1.5)
        assert mid is not None and mid > 0.3, (
            f'noise {noise:+.2f}: lone right boundary must centre LEFT of it, '
            f'got mid lateral {mid}')


def test_lone_boundary_no_memory_falls_back_to_sign():
    """Without history a clearly-offset lone boundary still sides by its sign."""
    lone = _line(0.8, 3.0, lambda x: -0.45)   # right boundary, robot centered
    out = centerline_from_line_cloud(lone, half_lane=HALF_LANE)
    mid = _mid_lateral(out, 1.5)
    assert mid is not None and abs(mid - 0.05) < 0.1


def test_x_disjoint_fragments_not_forced_opposite():
    """Two x-sequential fragments of ONE curving boundary (a U-turn apex split)
    must be offset to the SAME side. The old relative-order rule declared the
    far fragment the 'left boundary' and pulled the centerline between them."""
    # Right boundary curving left: y = 0.25*(x-1)^2 - 0.45, split with a gap.
    frag_near = _line(1.0, 1.9, lambda x: 0.25 * (x - 1.0) ** 2 - 0.45)
    frag_far = _line(2.7, 3.0, lambda x: 0.25 * (x - 1.0) ** 2 - 0.45)
    out = centerline_from_line_cloud(frag_near + frag_far, half_lane=HALF_LANE)
    # Same-side offset => every centre point sits LEFT of the boundary curve.
    for (x, y) in out[1:]:
        boundary_y = 0.25 * (x - 1.0) ** 2 - 0.45
        assert y > boundary_y, (x, y, boundary_y)


def test_staggered_opposite_boundaries():
    """On a curve the two boundaries are often visible over STAGGERED x-ranges
    (inner line near, outer line farther) with little or no x-overlap. They
    must still classify as LEFT/RIGHT — grouping them as one boundary offsets
    the far line a full lane the wrong way, and because the overlap jitters
    around zero, the resulting path jumped ~1.5 m between frames. The
    near-vertical junction between them (|dy|/dx >> lane slope) is what rules
    out the one-boundary reading."""
    left_near = _line(0.65, 1.30, lambda x: 0.45)
    right_far = _line(1.36, 2.60, lambda x: -0.32)
    out = centerline_from_line_cloud(left_near + right_far, half_lane=HALF_LANE)
    # Correct: left offsets right (~ -0.05), right offsets left (~ +0.18) —
    # everything near the lane centre. Wrong (grouped): far chain lands ~ -0.8.
    for (x, y) in out[1:]:
        assert abs(y) < 0.3, (x, y)


def test_near_equal_fragments_not_a_lane():
    """Two fragments whose laterals differ by centimetres cannot be opposite
    boundaries of a ~1 m lane; forcing them apart threw candidates +/-0.5 m."""
    a = _line(0.9, 1.4, lambda x: -0.15)
    b = _line(2.0, 2.6, lambda x: -0.18)
    out = centerline_from_line_cloud(a + b, half_lane=HALF_LANE)
    lats = [y for (x, y) in out[1:]]
    # One consistent side: all centre points on the same side of the fragments.
    assert all(y > -0.1 for y in lats) or all(y < -0.2 for y in lats), lats


def test_anchor_at_robot_origin():
    """The path must start exactly at the robot (0, 0): back-carrying the lane
    centre's lateral to x=0 (tried flat AND curvature-compensated, 2026-07-04)
    turned detection noise into a wandering e_lat the car chased — oscillation.
    Near-field offset information comes from camera coverage, not extrapolation."""
    curved = (_line(0.8, 1.6, lambda x: 0.3 * (x - 0.8) ** 2 + 0.5)
              + _line(0.8, 1.6, lambda x: 0.3 * (x - 0.8) ** 2 - 0.5))
    offset = _line(0.8, 3.0, lambda x: 0.7) + _line(0.8, 3.0, lambda x: -0.3)
    for cloud in (curved, offset):
        out = centerline_from_line_cloud(cloud, half_lane=HALF_LANE)
        assert out[0] == (0.0, 0.0), out[0]


def test_stray_short_chain_discarded():
    """A tiny far cluster (pixel-speckle survivor) between/beyond the boundaries
    must be dropped, not forced LEFT/RIGHT and offset half a lane sideways.
    Observed on real data (session 20260710_090357_depth_pair): a 4-point stray
    at y~0, x~4.2-4.6 kinked the centerline ~0.3-0.5 m laterally."""
    left = _line(1.0, 3.4, lambda x: 0.5)
    # Right boundary curves away after x=3.2 (like the real frame), leaving the
    # stray >link_gap from everything so it forms its own chain.
    right = _line(1.0, 4.6, lambda x: -0.5 - 0.8 * max(0.0, x - 3.2))
    stray = [(4.2, 0.0), (4.35, 0.1), (4.5, 0.05)]
    dbg = {}
    centerline_from_line_cloud(left + right + stray, half_lane=HALF_LANE,
                               debug=dbg)
    # Sanity: the stray really did cluster into its own chain upstream...
    assert len(dbg['chains_raw']) == 3, [len(c) for c in dbg['chains_raw']]
    # ...and the filter removed exactly it, keeping the two real boundaries.
    assert len(dbg['chains_kept']) == 2, [len(c) for c in dbg['chains_kept']]
    for cl in dbg['classified']:
        assert abs(cl['rep_y']) > 0.2, (
            f"mid-lane stray survived classification: rep_y={cl['rep_y']}")


def test_lone_short_chain_survives_stray_filter():
    """The stray filter must never discard the frame's ONLY chain: a lone short
    detection is still the only lane evidence available."""
    lone = [(1.0, -0.45), (1.2, -0.45), (1.4, -0.44)]
    out = centerline_from_line_cloud(lone, half_lane=HALF_LANE)
    assert len(out) >= 2, out


def test_path_folds_back():
    zigzag = [(0.0, 0.0), (1.0, 0.5), (0.8, -0.4), (1.8, 0.6)]
    assert path_folds_back(zigzag)
    # A tight but legitimate U-turn arc (R=1, ~17 deg/segment) must pass.
    arc = [(math.sin(t) * 1.0, 1.0 - math.cos(t) * 1.0)
           for t in [i * 0.3 for i in range(8)]]
    assert not path_folds_back(arc)
