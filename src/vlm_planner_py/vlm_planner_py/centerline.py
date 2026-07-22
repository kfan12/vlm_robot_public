"""Curve-aware centerline construction from metric cone positions.

Everything here works in the ROBOT (base_link) frame (x forward, y left) and is
ROS-free so it can be unit-tested without a node. The input is the unordered set
of cone positions produced by projection.project_pixels_to_base_link; the output
is an ordered centerline starting at the robot origin.

The key idea (see docs/detection_curve_tracking_plan.md): classify left/right
relative to the *running track heading*, not the camera image. Because the heading
updates after every gate, "left" and "right" are re-evaluated in the local frame,
so the classification flips correctly across an S-curve inflection.
"""

import math


def build_centerline_gatewalk(cones,
                              half_lane=0.4,
                              d_min=0.3,
                              d_max=3.0,
                              max_pts=20,
                              max_heading_step=math.pi / 4,
                              max_one_sided_lat=1.0):
    """Walk gates from the robot outward to build an ordered centerline.

    cones             : list of (x, y) cone positions in base_link (unordered).
    half_lane         : half the lane width [m]; used for one-sided gates and is
                        tied to the world's gate spacing (0.8 m -> 0.4).
    d_min, d_max      : forward search window for the next gate [m].
    max_pts           : safety cap on the number of centerline points.
    max_heading_step  : cap on how far the heading may turn per step [rad];
                        rejects outlier gates that would swing the path wildly.
    max_one_sided_lat : max lateral offset [m] a lone cone may have from the
                        running centerline to still count as a one-sided gate.
                        A real lane boundary sits ~half_lane off centre, so a
                        cone far beyond that is not a boundary of this gate
                        (e.g. a cone from a sharp upcoming curve seen across
                        the bend) and the walk stops rather than swinging out
                        to it.

    Returns: ordered list of (x, y) centerline points starting at (0, 0).
    """
    remaining = list(cones)
    pos = (0.0, 0.0)
    heading = 0.0
    centerline = [pos]

    while remaining and len(centerline) < max_pts:
        hx, hy = math.cos(heading), math.sin(heading)
        nx, ny = -hy, hx  # left normal of the heading

        # Forward distance and signed lateral offset (+ = left) of each cone
        # that lies within the forward search window.
        ahead = []
        for c in remaining:
            dx, dy = c[0] - pos[0], c[1] - pos[1]
            fwd = dx * hx + dy * hy
            lat = dx * nx + dy * ny
            if d_min <= fwd <= d_max:
                ahead.append((c, fwd, lat))
        if not ahead:
            break

        left = [t for t in ahead if t[2] > 0.0]
        right = [t for t in ahead if t[2] <= 0.0]
        l = min(left, key=lambda t: t[1]) if left else None   # nearest-forward left
        r = min(right, key=lambda t: t[1]) if right else None  # nearest-forward right

        if l and r:
            # Both boundaries present: midpoint is robust regardless of heading.
            lc, rc = l[0], r[0]
            gate = ((lc[0] + rc[0]) / 2.0, (lc[1] + rc[1]) / 2.0)
            consumed = (lc, rc)
        else:
            # One-sided gate. Do NOT trust the heading-relative sign to decide
            # which boundary this cone is: on a curve the heading lags the lane,
            # so a lone cone near the heading axis is easily mis-sided (which
            # offsets the centerline the wrong way). Instead compare the cone's
            # lateral position to the centerline SO FAR (pos): a boundary cone
            # sits ~half_lane off the local centre, and the lane centre drifts by
            # less than half_lane between gate steps, so the sign of (c_y - pos_y)
            # reliably tells left from right.
            cand_t = (l or r)
            cand, cand_lat = cand_t[0], cand_t[2]
            # Plausibility guard: a lone cone whose lateral offset from the
            # running centerline far exceeds half_lane is not a boundary of
            # this ~lane-width gate (it's typically a cone from a sharp
            # upcoming curve seen across the bend). Appending it would swing
            # the centerline metres sideways in a single step (and the raw
            # gate is appended regardless of max_heading_step, which only
            # clamps the *next* search heading). Stop the walk here so the
            # good prefix is kept instead of being corrupted/rejected.
            if abs(cand_lat) > max_one_sided_lat:
                break
            on_left = cand[1] >= pos[1]
            if on_left:
                # Left boundary -> centerline is half a lane to its RIGHT (-normal).
                gate = (cand[0] - nx * half_lane, cand[1] - ny * half_lane)
            else:
                # Right boundary -> centerline is half a lane to its LEFT (+normal).
                gate = (cand[0] + nx * half_lane, cand[1] + ny * half_lane)
            consumed = (cand,)

        # Heading toward this gate, with the per-step turn clamped so a far
        # off-axis gate can't swing the search direction past the cap.
        raw_heading = math.atan2(gate[1] - pos[1], gate[0] - pos[0])
        dtheta = math.atan2(math.sin(raw_heading - heading),
                            math.cos(raw_heading - heading))
        if dtheta > max_heading_step:
            dtheta = max_heading_step
        elif dtheta < -max_heading_step:
            dtheta = -max_heading_step

        centerline.append(gate)
        pos = gate
        heading = heading + dtheta
        # Always consume >=1 cone so the walk strictly progresses and terminates.
        remaining = [c for c in remaining if c not in consumed]

    return centerline


def centerline_from_line_cloud(points,
                               half_lane=0.5,
                               link_gap=0.6,
                               merge_bin=0.25,
                               resample_step=0.3,
                               grid_cell=0.1,
                               max_pts=24,
                               d_max=None,
                               side_memory=None,
                               debug=None):
    """Build a centerline from a dense cloud of CONTINUOUS-LINE boundary points.

    build_centerline_gatewalk is for sparse discrete cones and mis-pairs a near
    point on one line with a far point on the SAME line when fed a dense line
    cloud. This instead exploits that a road lane has CONSTANT width:

      1. Cluster the cloud into connected boundary CHAINS (nearest-neighbour
         linking with a gap threshold). Within the camera FOV + near range each
         boundary is x-monotonic, so a chain is ordered by forward distance.
      2. Classify each chain as the LEFT or RIGHT boundary ONCE, by the lateral
         sign of its near (smallest-x) end. Deciding per-chain, not per-slice,
         is what fixes the U-turn apex and the S-curve: a single boundary that
         curves across the heading axis keeps one consistent classification
         instead of flip-flopping.
      3. Offset every chain point half a lane toward the lane interior along the
         chain's local normal. For a constant-width lane this offset boundary IS
         the centerline (exact), whether one or both boundaries are visible.
      4. Merge the per-chain candidate centerlines (bin by forward distance,
         average laterally) into one ordered centerline.

    points : list of (x, y) line points in base_link (x fwd, y left), unordered.
    d_max  : if set, drop points farther than this forward distance [m]. Far
             markings project from noisy long-range depth, so their metric
             scatter drags the centerline off the lane; clipping the horizon to
             the reliable near range keeps the planned path inside the lane.
    side_memory : optional MUTABLE dict owned by the caller, carried across
             frames: {'left': y, 'right': y} = the last-seen near-end lateral of
             each boundary. Two-boundary frames refresh it; a SINGLE-boundary
             frame classifies its lone chain against the remembered lane (the
             midpoint of the stored laterals) instead of against y=0. Without
             this, a lone boundary near the robot's centreline (rep_y ~ 0) is
             sided by the NOISE SIGN of its lateral, and each flip throws the
             centerline a full lane width to the other side -> the path (and the
             car, once e_lat is coupled) oscillates. Pass {} on the first frame;
             pass None to disable (stateless legacy behaviour).
    Returns: ordered list of (x, y) centerline points starting at (0, 0).

    debug : if a dict is passed, every intermediate stage is recorded into it
            (input/gridded/chains_raw/chains_merged/chains_kept/
            chains_resampled/thresh/classified/cands/merged/final) for offline
            visualization by
            test/debug_centerline.py. None (default) => zero overhead.
    """
    if debug is not None:
        debug['input'] = [tuple(p) for p in points]
    # The real detector emits HUNDREDS of dense, noisy projected pixels per
    # boundary. Snap them to a coarse grid first: bounds the O(n^2) clustering
    # cost and pre-averages the worst pixel noise.
    pts = _grid_dedup([p for p in points
                       if p[0] > 0.0 and (d_max is None or p[0] <= d_max)],
                      grid_cell)
    if debug is not None:
        debug['gridded'] = list(pts)
    if len(pts) < 2:
        return [(0.0, 0.0)]

    chains = _cluster_chains(pts, link_gap)
    if debug is not None:
        debug['chains_raw'] = [list(c) for c in chains]
    # Reconnect fragments of ONE boundary that a depth gap / occlusion split
    # apart, before classifying. Without this a split boundary's far fragment is
    # classified by its own near end and can land on the wrong side (~1 m error).
    chains = _merge_collinear(chains)
    if debug is not None:
        debug['chains_merged'] = [list(c) for c in chains]
    # Drop stray fragments too small to be a trusted boundary. Far-range pixel
    # speckle that survives the detector's area gate projects to a few points
    # near mid-lane, forms a tiny chain _merge_collinear rightly refuses to
    # join, and would then be FORCED left/right below (a coin flip at
    # rep_y ~ thresh) and offset half a lane sideways — an artifact kink in the
    # merged centerline (observed on real sensor data, ~0.3-0.5 m). A real
    # boundary fragment has forward extent; a stray does not. Never drops the
    # last chain: a lone short chain is still the frame's only lane evidence.
    chains = _drop_stray_chains(chains)
    if debug is not None:
        debug['chains_kept'] = [list(c) for c in chains]
    # Resample each chain to a stable forward spacing, AVERAGING the points in
    # each slice. This is what kills the oscillation: tangents (and hence the
    # inward-offset normals) are then computed over a ~resample_step baseline
    # instead of between adjacent 2-3 cm noisy points, where atan2(noise, ~0)
    # makes the normal swing wildly and the offset boundary scatter.
    chains = [_resample_chain_x(sorted(c, key=lambda p: p[0]), resample_step)
              for c in chains]
    chains = [c for c in chains if len(c) >= 2]
    if debug is not None:
        debug['chains_resampled'] = [list(c) for c in chains]
    if not chains:
        return [(0.0, 0.0)]

    # Classify each chain LEFT (+1) or RIGHT (-1).
    #
    # With two boundaries visible, use their RELATIVE lateral order, not the
    # absolute sign of y: the chain with the greater mean-y is the left boundary,
    # the other the right. The decision threshold then sits BETWEEN the two lines
    # rather than at y=0. This is what keeps the centerline from collapsing onto a
    # line: absolute-sign classification fails when the robot hugs one line (that
    # line's points are at y~0, so its sign is noise) or when both boundaries fall
    # on the same side of the robot on a curve -- both then offset the same way and
    # the midpoint lands on a line. Relative order forces opposite sides, so the
    # midpoint is always strictly between the two boundaries.
    # Each chain's representative lateral = median y of its NEAREST 3 points. The
    # near end is reliable even when the boundary curves away far out (the whole-
    # chain mean would be dragged off by the curving tail, e.g. a U-turn apex).
    reps = [(_median([p[1] for p in c[:3]]), c) for c in chains]
    reps.sort(key=lambda m: m[0], reverse=True)        # left-most (max y) first

    # Are the outermost chains really the lane's TWO OPPOSITE boundaries?
    # Relative-order classification alone is not enough: fragments of ONE
    # boundary that _merge_collinear could not join (a U-turn apex bends too
    # sharply for its angle gate) also arrive as >=2 chains, and forcing them
    # onto opposite sides offsets them toward each other — the centerline then
    # jumps up to a full lane width when the fragments re-link a frame later
    # (the dominant path-oscillation source in the turn dumps). Gates:
    #   * separation: opposite boundaries sit ~2*half_lane apart laterally; two
    #     reps closer than half_lane cannot be opposite boundaries.
    #   * x-overlap: genuine left/right boundaries running ALONGSIDE each other
    #     (shared forward range) are opposite boundaries.
    #   * x-DISJOINT chains are NOT automatically one boundary: on curves the
    #     two boundaries are often visible over STAGGERED ranges (inner line
    #     near, outer line farther) with the overlap jittering around zero — a
    #     hard overlap>0 test then flickers between classification modes frame
    #     to frame and the path jumps a lane width (the residual "occasional
    #     shift"). Disambiguate by the JUNCTION SLOPE: if both chains were one
    #     continuous line, the bridge from the near chain's far end to the far
    #     chain's near start must have a lane-plausible slope (U-turn apex
    #     fragments measure <= ~2.3). A near-vertical junction (opposite
    #     boundaries measured 3.5-17 in the dumps) cannot be one line.
    two_boundaries = False
    if len(reps) >= 2 and (reps[0][0] - reps[-1][0]) > half_lane:
        ca, cb = reps[0][1], reps[-1][1]
        x_overlap = (min(ca[-1][0], cb[-1][0]) - max(ca[0][0], cb[0][0]))
        if x_overlap > 0.0:
            two_boundaries = True
        else:
            near_c, far_c = (ca, cb) if ca[0][0] <= cb[0][0] else (cb, ca)
            dy = abs(far_c[0][1] - near_c[-1][1])
            dx = max(far_c[0][0] - near_c[-1][0], 0.05)
            two_boundaries = (dy / dx) > 3.0

    if two_boundaries:
        # Threshold between the widest-separated boundary reps (top vs bottom) so
        # stray fragments are also placed relative to the lane, not to y=0.
        thresh = (reps[0][0] + reps[-1][0]) / 2.0
        sides = [1.0 if rep_y >= thresh else -1.0 for rep_y, _ in reps]
        if side_memory is not None:
            # Refresh the temporal side memory from this reliable two-boundary
            # frame: outermost reps are the lane's left/right boundaries.
            side_memory['left'] = reps[0][0]
            side_memory['right'] = reps[-1][0]
    else:
        # ONE boundary (possibly fragmented): every chain gets the SAME side.
        # Decide it from the fragment STARTING NEAREST the robot (the near end
        # is the reliable part), compared against the REMEMBERED lane midpoint
        # when history exists. The raw sign of y is only the last resort: a lone
        # boundary ridden at rep_y ~ 0 has a NOISE sign, and each wrong call
        # throws the centerline a full lane width to the other side (the
        # left/right flip-flop oscillation).
        if side_memory and 'left' in side_memory and 'right' in side_memory:
            thresh = (side_memory['left'] + side_memory['right']) / 2.0
        else:
            thresh = 0.0
        near_rep = min(reps, key=lambda m: m[1][0][0])[0]
        side_all = 1.0 if near_rep >= thresh else -1.0
        sides = [side_all] * len(reps)
        if side_memory is not None:
            # Keep the memory tracking through the single-boundary stretch:
            # store the seen side's near lateral and synthesize the unseen side
            # one lane width across (constant-width lane). A wrong call cannot
            # latch: the next two-boundary frame rewrites both entries.
            side_memory['left' if side_all > 0 else 'right'] = near_rep
            side_memory['right' if side_all > 0 else 'left'] = (
                near_rep - side_all * 2.0 * half_lane)

    if debug is not None:
        debug['thresh'] = thresh
        debug['classified'] = []
    cands = []
    for (rep_y, chain), side in zip(reps, sides):
        offset = _offset_chain_inward(chain, half_lane, side)
        cands.extend(offset)
        if debug is not None:
            # Keep chain<->offset paired so the plot can draw the inward-offset
            # arrows (chain point -> its centreline candidate) per boundary.
            debug['classified'].append({'side': side, 'rep_y': rep_y,
                                        'chain': list(chain),
                                        'offset': list(offset)})

    if not cands:
        return [(0.0, 0.0)]

    # Merge candidate centerlines: bin by forward distance, average laterally.
    cands.sort(key=lambda p: p[0])
    merged = []
    bx0, acc = cands[0][0], []
    for (x, y) in cands:
        if x - bx0 > merge_bin and acc:
            mx = sum(p[0] for p in acc) / len(acc)
            my = sum(p[1] for p in acc) / len(acc)
            merged.append((mx, my))
            bx0, acc = x, []
        acc.append((x, y))
    if acc:
        merged.append((sum(p[0] for p in acc) / len(acc),
                       sum(p[1] for p in acc) / len(acc)))

    # Anchor the path at the ROBOT ORIGIN (0, 0). Two revisions (2026-07-04)
    # tried anchoring at the lane centre's back-carried lateral instead, so the
    # controller would see a near-field cross-track error — BOTH oscillated in
    # sim and were reverted: with detection starting ~0.8-1.4 m out, any carry
    # back to x=0 (flat, tangent, or curvature-compensated) is speculation that
    # noise turns into a wandering e_lat the car then chases. The near-field
    # offset signal must come from actually SEEING the near lane (camera pitched
    # down so ground coverage starts ~0.5 m out), not from extrapolation: with
    # near coverage the first REAL centre points sit close to the robot and the
    # (0,0) stub carries almost no false information.
    out = ([(0.0, 0.0)] + merged)[:max_pts]
    if debug is not None:
        debug['cands'] = list(cands)
        debug['merged'] = list(merged)
        debug['final'] = list(out)
    return out


def path_folds_back(pts, max_turn_rad=math.pi / 2):
    """True when the polyline turns more than max_turn_rad between ADJACENT
    segments anywhere — i.e. it zigzags / doubles back on itself.

    A real lane centerline cannot do that: even the tight U-turn (R ~ 1 m) at
    ~0.3 m point spacing bends ~0.3/1.0 ~ 17 deg per segment, while the sawtooth
    the line-cloud builder produces when its x-monotonic assumption breaks (the
    U-turn apex seen sideways, or chains hopping between the two boundaries)
    turns 120-180 deg. Callers use this to REJECT such frames instead of
    publishing a path that slams the controller side to side."""
    for i in range(2, len(pts)):
        ax, ay = pts[i - 1][0] - pts[i - 2][0], pts[i - 1][1] - pts[i - 2][1]
        bx, by = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        if math.acos(cosang) > max_turn_rad:
            return True
    return False


def _grid_dedup(pts, cell):
    """Snap points to a `cell`-metre grid and average those landing in the same
    cell. Bounds the cloud size and pre-denoises before clustering."""
    if cell <= 0:
        return list(pts)
    buckets = {}
    for (x, y) in pts:
        k = (round(x / cell), round(y / cell))
        b = buckets.get(k)
        if b is None:
            buckets[k] = [x, y, 1]
        else:
            b[0] += x; b[1] += y; b[2] += 1
    return [(b[0] / b[2], b[1] / b[2]) for b in buckets.values()]


def _resample_chain_x(chain, step):
    """Average a chain into fixed forward (x) slices of width `step`. The chain
    must be x-sorted. Denoises laterally and yields ~evenly spaced points so the
    downstream tangent/normal estimate has a real baseline."""
    if len(chain) < 2 or step <= 0:
        return list(chain)
    out = []
    x0 = chain[0][0]
    sx = sy = 0.0
    n = 0
    for (x, y) in chain:
        if x - x0 > step and n > 0:
            out.append((sx / n, sy / n))
            x0 = x
            sx = sy = 0.0
            n = 0
        sx += x; sy += y; n += 1
    if n > 0:
        out.append((sx / n, sy / n))
    return out


def _cluster_chains(pts, link_gap):
    """Greedy nearest-neighbour clustering into connected chains. Seeds each
    chain at the unused point nearest the robot, then extends to the nearest
    unused point within link_gap."""
    remaining = list(pts)
    chains = []
    while remaining:
        seed = min(remaining, key=lambda p: p[0] * p[0] + p[1] * p[1])
        remaining.remove(seed)
        chain = [seed]
        grew = True
        while grew:
            grew = False
            tail = chain[-1]
            nxt, nd = None, link_gap
            for p in remaining:
                d = math.hypot(p[0] - tail[0], p[1] - tail[1])
                if d < nd:
                    nd, nxt = d, p
            if nxt is not None:
                chain.append(nxt)
                remaining.remove(nxt)
                grew = True
        chains.append(chain)
    return chains


def _merge_collinear(chains, max_gap=1.6, max_angle=0.7):
    """Join chains that are fragments of the SAME boundary (one continues the
    other). A fragment pair qualifies when the far end of A is within max_gap of
    the near end of B AND the join is roughly straight: A's far tangent, the
    join vector, and B's near tangent are mutually within max_angle. The two
    OPPOSITE boundaries do NOT merge -- they run parallel, so the join vector
    between them is ~lateral (a large angle) and is rejected."""
    chains = [sorted(c, key=lambda p: p[0]) for c in chains if len(c) >= 2]
    changed = True
    while changed and len(chains) > 1:
        changed = False
        best = None
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i == j:
                    continue
                A, B = chains[i], chains[j]
                gap = math.hypot(B[0][0] - A[-1][0], B[0][1] - A[-1][1])
                if gap > max_gap:
                    continue
                ta = _unit(A[-1][0] - A[-2][0], A[-1][1] - A[-2][1])
                tb = _unit(B[1][0] - B[0][0], B[1][1] - B[0][1])
                jv = _unit(B[0][0] - A[-1][0], B[0][1] - A[-1][1])
                a1, a2 = _angle(ta, jv), _angle(jv, tb)
                if a1 < max_angle and a2 < max_angle:
                    score = gap + a1 + a2
                    if best is None or score < best[0]:
                        best = (score, i, j)
        if best is not None:
            _, i, j = best
            chains[i] = sorted(chains[i] + chains[j], key=lambda p: p[0])
            chains.pop(j)
            changed = True
    return chains


def _drop_stray_chains(chains, min_pts=3, min_extent=0.5):
    """Discard chains too small to be trusted as a real boundary fragment:
    fewer than min_pts points or under min_extent forward extent [m]. Chains
    must be x-sorted (as _merge_collinear returns them). A single chain passes
    through untouched, and if EVERY chain fails the gate the longest-extent one
    is kept — non-empty input never filters to empty."""
    if len(chains) <= 1:
        return chains
    kept = [c for c in chains
            if len(c) >= min_pts and c[-1][0] - c[0][0] >= min_extent]
    if not kept:
        kept = [max(chains, key=lambda c: c[-1][0] - c[0][0])]
    return kept


def _unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def _angle(u, v):
    d = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.acos(d)


def _offset_chain_inward(chain, half_lane, side):
    """Offset each chain point half a lane toward the lane interior along the
    local normal. side=+1 for a left boundary (interior to its right), -1 for a
    right boundary (interior to its left)."""
    out = []
    n = len(chain)
    for i in range(n):
        a = chain[max(0, i - 1)]
        b = chain[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        tn = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tn, ty / tn
        # Left boundary -> inward is the RIGHT normal (ty, -tx); right boundary
        # -> inward is the LEFT normal (-ty, tx).
        inx, iny = (ty, -tx) if side > 0 else (-ty, tx)
        out.append((chain[i][0] + inx * half_lane,
                    chain[i][1] + iny * half_lane))
    return out


def bias_path_inward(pts, side, shift_m, margin, half_lane, ramp_m=1.0):
    """Shift the path toward the TURN INTERIOR by a smooth arc-length RAMP: 0 at the near
    end (the robot), easing (smoothstep) to the target offset by `ramp_m` of arc length,
    then held. This biases the racing line to the inside of the curve for left/right
    maneuvers. Mirrors the _offset_chain_inward normal-offset pattern.

    Two design choices, both to fix the earlier drift/jaggedness:
      * Ramp from 0 at pose 0 -- the path near end stays AT the robot, so the MPC sees no
        standing cross-track error (a shifted pose 0 = a permanent e_lat the controller
        steers into, which walked the car off the line -- the §4.2 line-hugging failure).
      * FIXED offset, NOT curvature-proportional -- the curve is not in view when the
        maneuver commits, so curvature buys ~nothing here (plan R-C) and a per-point
        curvature estimate is the main jitter source. A smooth ramp on a smooth centerline
        offset stays smooth by construction.

    side = +1 (LEFT turn) shifts along the LEFT normal (toward +y for a forward path);
    side = -1 (RIGHT turn) along the RIGHT normal. NOTE this is the OPPOSITE meaning of
    _offset_chain_inward's `side` (there +1 = a left BOUNDARY, offset to its RIGHT). The
    target offset = min(shift_m, half_lane - margin) so it never crosses within `margin`
    of the inside line. Returns a new list; the input is unchanged. shift_m <= 0 or
    < 3 points -> no bias (unchanged copy)."""
    n = len(pts)
    if n < 3 or shift_m <= 0.0:
        return list(pts)
    target = min(shift_m, max(0.0, half_lane - margin))
    out = []
    s = 0.0
    for i in range(n):
        if i > 0:
            s += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        # Smoothstep ease-in 0->1 over ramp_m, held at 1 after (so pose 0 shift = 0).
        u = min(1.0, s / ramp_m) if ramp_m > 1e-6 else 1.0
        shift = target * (u * u * (3.0 - 2.0 * u))
        # Local tangent (central diff on the already-smoothed centerline).
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        tn = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tn, ty / tn
        # Left turn -> LEFT normal (-ty, tx); right turn -> RIGHT normal (ty, -tx).
        nx, ny = (-ty, tx) if side > 0 else (ty, -tx)
        out.append((pts[i][0] + nx * shift, pts[i][1] + ny * shift))
    return out


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0
