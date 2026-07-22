#!/usr/bin/env python3
"""Analyze a recorded debugkit session: plots + metrics + report.

Reads a session folder produced by record_session.py and writes into
<session>/analysis/:

    trajectory.png        XY overview: truth vs estimates, path snapshots, signs
    odom_drift.png        position/yaw error of /odom and /odom_ekf vs truth
    oscillation.png       lateral wobble + steering activity on straight segments
    signs_timeline.png    sign read vs pending (armed) sign vs maneuver state vs speed
    speed_tracking.png    target vs actual speed + tracking error
    tracking_error.png    cross-track distance to the active planner path
    report.md             human-readable summary of everything below
    metrics.json          all computed metrics, machine-readable

No ROS required — pure numpy + matplotlib on the saved CSVs, so it runs
anywhere (system python or the venv):

    scripts/analyze_session.sh latest
    scripts/analyze_session.sh 20260705_141230_baseline --only drift,oscillation
    python3 tools/debugkit/analyze_session.py data/sessions/<name>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seslib as sl  # noqa: E402

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_SESSIONS = os.path.join(WS_ROOT, 'data', 'sessions')
WORLD_SDF = os.path.join(WS_ROOT, 'src', 'robotcar_gazebo', 'worlds',
                          'track_lines.world.sdf')

ACTIONABLE = ('left', 'right', 'winding', 'stop')


# ---------------------------------------------------------------------------
# Session context: lazy signal access + shared references
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, session):
        self.session = session
        self.out = os.path.join(session, 'analysis')
        os.makedirs(self.out, exist_ok=True)
        self.meta = sl.load_meta(session)
        self._cache = {}
        self.t0 = self._find_t0()

    def sig(self, name):
        if name not in self._cache:
            s = sl.load_signal(self.session, name)
            # /vlm/sign moved to a JSON payload ('{"sign": "left", "stamp": ..}');
            # analyses want the bare label, so unwrap old and new format here
            if s is not None and name == 'sign' and len(s.get('value', [])):
                s['value'] = np.array(
                    [sl.sign_label(v) for v in s['value']], dtype=object)
            self._cache[name] = s
        return self._cache[name]

    def _find_t0(self):
        t0 = None
        for name in ('odom_truth', 'odom_ekf', 'odom', 'cmd_vel', 'imu'):
            s = sl.load_signal(self.session, name)
            self._cache[name] = s
            if s is not None and len(s['t']):
                t = float(s['t'][0])
                t0 = t if t0 is None else min(t0, t)
        return t0 or 0.0

    def pose_ref(self):
        """Best available pose source: truth > ekf > wheel odom."""
        for name in ('odom_truth', 'odom_ekf', 'odom'):
            s = self.sig(name)
            if s is not None and len(s['t']) > 10:
                return name, s
        return None, None

    def state_segments(self):
        """[(label, t_start, t_stop)] from /maneuver/state, or []."""
        st = self.sig('maneuver_state')
        if st is None or len(st['t']) < 1:
            return []
        _, ref = self.pose_ref()
        t_end = float(ref['t'][-1]) if ref is not None else float(st['t'][-1])
        return sl.label_segments(st['t'], st['value'], t_end=t_end)

    def straight_segments(self, min_duration=3.0):
        """Straight-driving windows: /maneuver/state=='straight' if recorded,
        else low-yaw-rate windows on the reference odom."""
        segs = [(a, b) for lab, a, b in self.state_segments()
                if lab == 'straight' and (b - a) >= min_duration]
        if segs:
            return segs, 'maneuver_state == straight'
        name, ref = self.pose_ref()
        if ref is None:
            return [], 'no pose source'
        t, wz = ref['t'], ref['wz']
        k = max(1, int(round(0.5 / max(np.median(np.diff(t)), 1e-3))))
        wz_s = np.convolve(wz, np.ones(k) / k, mode='same')
        straight = np.abs(wz_s) < 0.06
        segs, start = [], None
        for i in range(len(t)):
            if straight[i] and start is None:
                start = t[i]
            elif not straight[i] and start is not None:
                if t[i] - start >= min_duration:
                    segs.append((float(start), float(t[i])))
                start = None
        if start is not None and t[-1] - start >= min_duration:
            segs.append((float(start), float(t[-1])))
        return segs, f'|yaw rate| < 0.06 rad/s on {name} (no maneuver_state recorded)'


def shade_states(ax, ctx, alpha=0.55):
    """Background bands for non-straight maneuver states."""
    for lab, a, b in ctx.state_segments():
        c = sl.STATE_COLORS.get(lab, '#ffffff')
        if c != '#ffffff':
            ax.axvspan(a - ctx.t0, b - ctx.t0, color=c, alpha=alpha, lw=0, zorder=0)


# ---------------------------------------------------------------------------
# Sections — each returns (metrics_dict_or_None, list_of_report_lines)
# ---------------------------------------------------------------------------

def sec_overview(ctx, plt):
    name, ref = ctx.pose_ref()
    if ref is None:
        return None, ['skipped: no odometry recorded']
    fig, ax = plt.subplots(figsize=(7.5, 6))
    snaps = sl.load_path_snapshots(ctx.session, 'vlm_path', max_snapshots=40)
    for i, (_, poly) in enumerate(snaps):
        ax.plot(poly[:, 0], poly[:, 1], color=sl.COLORS['path'], alpha=0.18,
                lw=0.8, zorder=1,
                label='planner path snapshots' if i == 0 else None)

    # /odom_truth lives in Gazebo's WORLD frame (origin at wherever the world
    # file puts it), while /odom and /odom_ekf -- and the path snapshots
    # plotted above -- are zeroed in their own local "odom" frame. That's a
    # constant translation, not drift (see the matching alignment in
    # sec_drift), but left alone it makes the ground-truth line sit a fixed
    # offset away from everything else in this plot. Shift ground truth into
    # the local frame instead of leaving it in world coordinates.
    local = ctx.sig('odom_ekf') or ctx.sig('odom')
    truth = ctx.sig('odom_truth')
    offset = None
    if truth is not None and local is not None:
        offset = (truth['x'][0] - local['x'][0], truth['y'][0] - local['y'][0])

    for nm, label in (('odom_truth', 'ground truth'), ('odom_ekf', 'EKF'),
                      ('odom', 'wheel odom')):
        s = ctx.sig(nm)
        if s is not None:
            x, y = s['x'], s['y']
            if nm == 'odom_truth' and offset is not None:
                x = x - offset[0]
                y = y - offset[1]
            ls = '--' if nm == 'odom_truth' else '-'
            ax.plot(x, y, ls, color=sl.COLORS[nm], label=label, zorder=3)

    # `ref` (start marker + sign positions) may itself be odom_truth -- shift
    # it the same way so it lines up with the plotted (now local-frame) line.
    ref_x, ref_y = ref['x'], ref['y']
    if name == 'odom_truth' and offset is not None:
        ref_x = ref_x - offset[0]
        ref_y = ref_y - offset[1]

    ax.plot(ref_x[0], ref_y[0], 'o', color=sl.INK, ms=6, zorder=5)
    ax.annotate('start', (ref_x[0], ref_y[0]), textcoords='offset points',
                xytext=(6, 6), fontsize=8, color=sl.INK2)

    # Actual sign-board positions from the world file (world frame -- same
    # shift as ground truth). Distinct from the VLM sign markers below: these
    # are where the boards physically are, not where/when the robot read one.
    if offset is not None:
        world_signs = sl.load_world_signs(WORLD_SDF)
        for i, (lab, (wx, wy)) in enumerate(world_signs.items()):
            bx, by = wx - offset[0], wy - offset[1]
            ax.plot(bx, by, '*', color=sl.MUTED, ms=14, mew=0, zorder=4,
                    label='sign board (actual)' if i == 0 else None)
            ax.annotate(lab, (bx, by), textcoords='offset points',
                        xytext=(7, 4), fontsize=8, color=sl.MUTED,
                        style='italic')
    # Mark the arbitrated/confirmed candidate (/vlm/pending_sign) rather than
    # every raw VLM read (/vlm/sign): the raw stream re-fires on each query of
    # the same still-visible board, so it puts a cluster of near-duplicate
    # marks at (almost) one spot; pending_sign only changes once a candidate
    # is actually armed, giving one mark per real event. Falls back to the
    # raw read for older sessions recorded without the pending_sign signal.
    # `actionable_sign_reads` below still counts raw /vlm/sign reads
    # regardless of which one is plotted -- it feeds sign_response.yaml's
    # scenario gate (metrics_catalog.yaml), which must keep its original
    # meaning independent of this display choice.
    pend = ctx.sig('pending_sign')
    sign = ctx.sig('sign')
    use_pending = pend is not None and len(pend['t']) > 0
    marker_sig = pend if use_pending else sign
    marker_color = sl.COLORS['pending'] if use_pending else sl.COLORS['sign']
    marker_label = 'confirmed sign' if use_pending else 'sign read'
    n_signs = 0
    if marker_sig is not None:
        te, le = sl.dedup_events(marker_sig['t'], marker_sig['value'])
        for t, lab in zip(te, le):
            if str(lab) not in ACTIONABLE:
                continue
            sx = sl.interp(t, ref['t'], ref_x)
            sy = sl.interp(t, ref['t'], ref_y)
            ax.plot(sx, sy, 'x', color=marker_color, ms=9, mew=2, zorder=6,
                    label=marker_label if n_signs == 0 else None)
            ax.annotate(str(lab), (sx, sy), textcoords='offset points',
                        xytext=(6, -10), fontsize=8, color=marker_color)
            n_signs += 1
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory overview')
    ax.legend(loc='best', fontsize=8)
    fig.savefig(os.path.join(ctx.out, 'trajectory.png'))
    plt.close(fig)
    dist = sl.cumdist(ref['x'], ref['y'])[-1]
    n_actionable_reads = 0
    if sign is not None:
        _, sle = sl.dedup_events(sign['t'], sign['value'])
        n_actionable_reads = sum(1 for lab in sle if str(lab) in ACTIONABLE)
    m = {'pose_source': name, 'distance_m': round(float(dist), 2),
         'duration_s': round(float(ref['t'][-1] - ref['t'][0]), 1),
         'actionable_sign_reads': n_actionable_reads}
    return m, [f"pose source `{name}`, {m['distance_m']} m driven in "
               f"{m['duration_s']} s, {n_actionable_reads} actionable sign "
               f"read(s) ({n_signs} {marker_label} marker(s) plotted)",
               '![trajectory](trajectory.png)']


def sec_drift(ctx, plt):
    truth = ctx.sig('odom_truth')
    if truth is None or len(truth['t']) < 10:
        return None, ['skipped: /odom_truth not recorded (needed as reference)']
    tt = truth['t']
    dist = sl.cumdist(truth['x'], truth['y'])
    yaw_t = sl.unwrap(truth['yaw'])
    ests = [(n, ctx.sig(n)) for n in ('odom', 'odom_ekf')]
    ests = [(n, s) for n, s in ests if s is not None and len(s['t']) > 10]
    if not ests:
        return None, ['skipped: no estimate (/odom or /odom_ekf) recorded']

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=False)
    metrics, lines = {}, []
    for name, s in ests:
        ex = sl.interp(tt, s['t'], s['x'])
        ey = sl.interp(tt, s['t'], s['y'])
        eyaw = sl.interp(tt, s['t'], sl.unwrap(s['yaw']))
        # Align origins at the first sample: /odom_truth is in WORLD frame
        # (spawn at x=-1.5) while the estimates start at (0,0), so without
        # this the constant spawn offset masquerades as ~1.5 m of "drift"
        # (autotune loops 0-8 all showed it). Drift = error GROWTH from the
        # aligned start, not absolute frame offset.
        ex = ex - (ex[0] - truth['x'][0])
        ey = ey - (ey[0] - truth['y'][0])
        perr = np.hypot(ex - truth['x'], ey - truth['y'])
        yerr = np.degrees(np.abs(sl.ang_diff(eyaw, yaw_t)))
        c = sl.COLORS[name]
        axes[0].plot(tt - ctx.t0, perr, color=c, label=name)
        axes[1].plot(tt - ctx.t0, yerr, color=c, label=name)
        axes[2].plot(dist, perr, color=c, label=name)
        m = {'pos_rmse_m': sl.rmse(perr),
             'pos_max_m': float(perr.max()),
             'pos_final_m': float(perr[-1]),
             'yaw_rmse_deg': sl.rmse(yerr),
             'drift_pct_of_dist': float(perr[-1] / max(dist[-1], 1e-6) * 100.0)}
        metrics[name] = {k: round(v, 4) for k, v in m.items()}
        lines.append(f"`{name}` vs truth: RMSE {m['pos_rmse_m']:.3f} m, "
                     f"final {m['pos_final_m']:.3f} m "
                     f"({m['drift_pct_of_dist']:.2f}% of {dist[-1]:.1f} m), "
                     f"yaw RMSE {m['yaw_rmse_deg']:.2f} deg")
    for ax in axes[:2]:          # panel 3 is over distance, not time
        shade_states(ax, ctx)
    axes[0].set_ylabel('position error [m]')
    axes[0].set_title('Odometry drift vs ground truth')
    axes[1].set_ylabel('|yaw error| [deg]')
    axes[1].set_xlabel('t [s]')
    axes[2].set_ylabel('position error [m]')
    axes[2].set_xlabel('distance traveled [m]')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.out, 'odom_drift.png'))
    plt.close(fig)
    lines.append('![odom drift](odom_drift.png)')
    return metrics, lines


def sec_oscillation(ctx, plt, min_seg=3.0):
    name, ref = ctx.pose_ref()
    if ref is None:
        return None, ['skipped: no odometry recorded']
    segs, how = ctx.straight_segments(min_seg)
    if not segs:
        return None, [f'skipped: no straight segment >= {min_seg:.0f}s found ({how})']
    cmd = ctx.sig('cmd_vel')

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8))
    metrics, lines = {'segments': [], 'detection': how, 'pose_source': name}, []
    longest = None
    for i, (a, b) in enumerate(segs):
        msk = (ref['t'] >= a) & (ref['t'] <= b)
        if msk.sum() < 20:
            continue
        # Trim to the longest contiguous CRUISE window (|vx| > 0.3): straight
        # *state* segments include standing (planner gate wait) and creep
        # phases, which flatten the PCA line fit and inflate lateral std with
        # non-control motion. Oscillation is only meaningful while tracking
        # at speed.
        if 'vx' in ref:
            sub_idx = np.flatnonzero(msk)
            mov = np.flatnonzero(np.abs(ref['vx'][sub_idx]) > 0.3)
            if mov.size < 20:
                continue
            runs = np.split(mov, np.flatnonzero(np.diff(mov) > 25) + 1)
            best = max(runs, key=len)
            if best.size < 20:
                continue
            msk = np.zeros_like(msk)
            msk[sub_idx[best[0]]:sub_idx[best[-1]] + 1] = True
            a = float(ref['t'][sub_idx[best[0]]])
            b = float(ref['t'][sub_idx[best[-1]]])
            if b - a < min_seg:
                continue
        t = ref['t'][msk]
        along, lat = sl.line_fit_lateral(ref['x'][msk], ref['y'][msk])
        f_lat, _, _ = sl.dominant_freq(t, lat)
        seg_m = {'t_start_s': round(float(a - ctx.t0), 1),
                 'duration_s': round(float(b - a), 1),
                 'length_m': round(float(along.max() - along.min()), 2),
                 'lateral_std_m': round(float(np.std(lat)), 4),
                 'lateral_p2p_m': round(float(lat.max() - lat.min()), 4),
                 'lateral_dom_freq_hz': round(f_lat, 2) if f_lat else None}
        if cmd is not None:
            cm = (cmd['t'] >= a) & (cmd['t'] <= b)
            if cm.sum() > 20:
                f_st, _, _ = sl.dominant_freq(cmd['t'][cm], cmd['wz'][cm])
                seg_m['steer_cmd_std'] = round(float(np.std(cmd['wz'][cm])), 4)
                seg_m['steer_dom_freq_hz'] = round(f_st, 2) if f_st else None
        metrics['segments'].append(seg_m)
        axes[0].plot(t - ctx.t0, lat, color=sl.COLORS['actual'], lw=1.2)
        axes[0].axvspan(a - ctx.t0, b - ctx.t0, color='#f0efec', zorder=0)
        if longest is None or (b - a) > (longest[1] - longest[0]):
            longest = (a, b, t, lat)
        lines.append(
            f"segment {i + 1} (t={seg_m['t_start_s']}s, {seg_m['duration_s']}s, "
            f"{seg_m['length_m']} m): lateral std {seg_m['lateral_std_m'] * 100:.1f} cm, "
            f"p2p {seg_m['lateral_p2p_m'] * 100:.1f} cm"
            + (f", dom. freq {seg_m['lateral_dom_freq_hz']} Hz"
               if seg_m['lateral_dom_freq_hz'] else ''))
    if not metrics['segments']:
        plt.close(fig)
        return None, ['skipped: straight segments too short for analysis']
    axes[0].set_ylabel('lateral deviation [m]')
    axes[0].set_xlabel('t [s]')
    axes[0].set_title(f'Straight-segment oscillation  ({how})')

    if cmd is not None:
        axes[1].plot(cmd['t'] - ctx.t0, cmd['wz'], color=sl.COLORS['cmd'],
                     lw=1.0, label='cmd_vel angular.z')
        for a, b in segs:
            axes[1].axvspan(a - ctx.t0, b - ctx.t0, color='#f0efec', zorder=0)
        axes[1].legend(fontsize=8, loc='upper right')
    axes[1].set_ylabel('steer command [rad/s]')
    axes[1].set_xlabel('t [s]')

    a, b, t, lat = longest
    f, freqs, amp = sl.dominant_freq(t, lat)
    if freqs is not None:
        axes[2].plot(freqs, amp * 100, color=sl.COLORS['actual'],
                     label='lateral deviation')
        if f:
            axes[2].axvline(f, color=sl.MUTED, lw=0.8, ls=':')
            axes[2].annotate(f'{f:.2f} Hz', (f, amp.max() * 100), fontsize=8,
                             color=sl.INK2, xytext=(4, -2),
                             textcoords='offset points')
    if cmd is not None:
        cm = (cmd['t'] >= a) & (cmd['t'] <= b)
        if cm.sum() > 20:
            _, fq2, am2 = sl.dominant_freq(cmd['t'][cm], cmd['wz'][cm])
            if fq2 is not None:
                ax2max = max(np.max(amp * 100), 1e-9)
                axes[2].plot(fq2, am2 / max(am2.max(), 1e-9) * ax2max,
                             color=sl.COLORS['cmd'], lw=1.0,
                             label='steer cmd (scaled)')
    axes[2].set_xlim(0, 3.0)
    axes[2].set_xlabel('frequency [Hz]')
    axes[2].set_ylabel('amplitude [cm]')
    axes[2].set_title('Spectrum of longest straight segment')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.out, 'oscillation.png'))
    plt.close(fig)
    lines.append('![oscillation](oscillation.png)')
    worst = max(metrics['segments'], key=lambda s: s['lateral_std_m'])
    metrics['worst_lateral_std_m'] = worst['lateral_std_m']
    return metrics, lines


def sec_signs(ctx, plt):
    sign = ctx.sig('sign')
    if sign is None or len(sign['t']) == 0:
        return None, ['skipped: /vlm/sign not recorded']
    st = ctx.sig('maneuver_state')
    pend = ctx.sig('pending_sign')
    tse, sle = sl.dedup_events(sign['t'], sign['value'])

    reads = [{'t_s': round(float(t - ctx.t0), 1), 'label': str(l)}
             for t, l in zip(tse, sle)]
    counts = {}
    for v in sign['value']:
        counts[str(v)] = counts.get(str(v), 0) + 1

    pte, ptl = (sl.dedup_events(pend['t'], pend['value'])
                if pend is not None and len(pend['t']) else ([], []))
    ste, stl = (sl.dedup_events(st['t'], st['value'])
                if st is not None and len(st['t']) else ([], []))

    def first_hit(events_t, events_l, lab, after_t):
        # First event of `lab` at/after after_t, within a 60s window -- mirrors
        # the pre-existing engage-latency window so arm/engage are comparable.
        hit = [ts for ts, l_ in zip(events_t, events_l)
               if str(l_) == lab and ts >= after_t and ts - after_t < 60.0]
        return round(float(hit[0] - after_t), 2) if hit else None

    latencies = []
    for t, lab in zip(tse, sle):
        lab = str(lab)
        if lab not in ACTIONABLE:
            continue
        entry = {'t_s': round(float(t - ctx.t0), 1), 'label': lab}
        if pend is not None:
            # Read -> ARMED: how long until the arbitrated candidate (distance-
            # gated: also requires the board within sign_reach_dist_m) matches
            # this read's label. Can be None even for a correct read that was
            # too far out when captured (see _arbitrate_pending_sign).
            entry['arm_latency_s'] = first_hit(pte, ptl, lab, t)
        if st is not None:
            entry['engage_latency_s'] = first_hit(ste, stl, lab, t)
        latencies.append(entry)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9), sharex=True)
    labels = sorted({str(v) for v in sign['value']})
    ymap = {l: i for i, l in enumerate(labels)}
    axes[0].scatter(sign['t'] - ctx.t0, [ymap[str(v)] for v in sign['value']],
                    s=14, color=sl.COLORS['sign'], zorder=3)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel('/vlm/sign\n(raw read)')
    axes[0].set_title('VLM sign read vs arbitrated candidate vs maneuver state vs speed')

    if pend is not None and len(pend['t']):
        plabels = sorted({str(v) for v in pend['value']})
        pmap = {l: i for i, l in enumerate(plabels)}
        axes[1].step(pend['t'] - ctx.t0, [pmap[str(v)] for v in pend['value']],
                     where='post', color=sl.COLORS['pending'])
        axes[1].set_yticks(range(len(plabels)))
        axes[1].set_yticklabels(plabels)
    axes[1].set_ylabel('/vlm/pending_sign\n(armed candidate)')

    if st is not None and len(st['t']):
        slabels = sorted({str(v) for v in st['value']})
        smap = {l: i for i, l in enumerate(slabels)}
        axes[2].step(st['t'] - ctx.t0, [smap[str(v)] for v in st['value']],
                     where='post', color=sl.COLORS['path'])
        axes[2].set_yticks(range(len(slabels)))
        axes[2].set_yticklabels(slabels)
    axes[2].set_ylabel('/maneuver/state\n(committed)')

    tgt = ctx.sig('target_speed')
    if tgt is not None:
        axes[3].step(tgt['t'] - ctx.t0, tgt['value'], where='post',
                     color=sl.COLORS['cmd'], label='target speed')
    _, ref = ctx.pose_ref()
    if ref is not None:
        axes[3].plot(ref['t'] - ctx.t0, ref['vx'], color=sl.COLORS['actual'],
                     lw=1.0, label='actual speed')
    axes[3].set_ylabel('speed [m/s]')
    axes[3].set_xlabel('t [s]')
    axes[3].legend(fontsize=8)
    for ax in axes:
        shade_states(ax, ctx)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.out, 'signs_timeline.png'))
    plt.close(fig)

    m = {'reads': reads, 'label_message_counts': counts, 'engagements': latencies}
    lines = [f"{len(reads)} sign transitions; message counts: "
             + ', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))]
    if pend is None:
        lines.append('note: /vlm/pending_sign not recorded (older session, or '
                      '"pending_sign" missing from the record preset) -- arm '
                      'latency unavailable, panel left blank')
    for e in latencies:
        stage = f"sign `{e['label']}` at t={e['t_s']}s"
        if 'arm_latency_s' in e:
            stage += (f" -> armed after {e['arm_latency_s']}s"
                      if e['arm_latency_s'] is not None
                      else ' -> NEVER armed within 60s')
        if 'engage_latency_s' in e:
            stage += (f" -> state engaged after {e['engage_latency_s']}s"
                      if e['engage_latency_s'] is not None
                      else ' -> NO matching state engagement within 60s')
        lines.append(stage)
    lines.append('![signs timeline](signs_timeline.png)')
    return m, lines


def sec_speed(ctx, plt):
    tgt = ctx.sig('target_speed')
    vref = ctx.sig('mpc_vref')
    _, ref = ctx.pose_ref()
    if tgt is None or ref is None or len(tgt['t']) < 2:
        return None, ['skipped: needs /maneuver/target_speed + an odometry source']

    # Compare actual speed against the slew-limited reference the QP actually
    # chases (/mpc/v_ref_ramped), not the raw stepwise setpoint -- the setpoint
    # jumps instantly on a maneuver change, so scoring against it counts the
    # (intentional) ramp-up/down lag as tracking error. Falls back to the step
    # setpoint for older sessions recorded without the `debug` preset.
    use_ramp = vref is not None and len(vref['t']) >= 2
    ref_sig = vref if use_ramp else tgt
    t = ref['t']
    msk = (t >= ref_sig['t'][0]) & (t <= ref_sig['t'][-1])
    if msk.sum() < 10:
        return None, ['skipped: no time overlap between speed reference and odometry']
    t = t[msk]
    actual = ref['vx'][msk]
    compare = sl.interp(t, ref_sig['t'], ref_sig['value']) if use_ramp \
        else sl.step_interp(t, ref_sig['t'], ref_sig['value'])
    err = actual - compare
    moving = compare > 0.05

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].step(tgt['t'] - ctx.t0, tgt['value'], where='post',
                 color=sl.COLORS['cmd'], lw=1.0, label='setpoint')
    if use_ramp:
        axes[0].step(vref['t'] - ctx.t0, vref['value'], where='post',
                     color='green', lw=1.3, label='ramped reference')
    axes[0].plot(t - ctx.t0, actual, color=sl.COLORS['actual'], lw=1.1,
                 label='actual')
    axes[0].set_ylabel('speed [m/s]')
    axes[0].set_title('Speed tracking' + (' (vs. ramped reference)' if use_ramp
                                           else ' (vs. stepwise setpoint)'))
    axes[0].legend(fontsize=8)
    axes[1].plot(t - ctx.t0, err, color=sl.COLORS['accent'], lw=1.0)
    axes[1].axhline(0, color=sl.MUTED, lw=0.8)
    axes[1].set_ylabel('error [m/s]')
    axes[1].set_xlabel('t [s]')
    for ax in axes:
        shade_states(ax, ctx)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.out, 'speed_tracking.png'))
    plt.close(fig)

    m = {'rmse_mps': round(sl.rmse(err[moving]), 4) if moving.any() else None,
         'mean_err_mps': round(float(err[moving].mean()), 4) if moving.any() else None,
         'max_abs_err_mps': round(float(np.abs(err).max()), 4),
         'compared_to': 'ramped_reference' if use_ramp else 'stepwise_setpoint'}
    against = 'ramped reference' if use_ramp else 'stepwise setpoint (no mpc_vref recorded)'
    return m, [f"speed tracking vs {against} (while moving > 0.05 m/s): "
               f"RMSE {m['rmse_mps']} m/s, mean {m['mean_err_mps']} m/s, "
               f"max |err| {m['max_abs_err_mps']} m/s",
               '![speed tracking](speed_tracking.png)']


def sec_tracking(ctx, plt):
    snaps = sl.load_path_snapshots(ctx.session, 'vlm_path')
    if not snaps:
        return None, ["skipped: no vlm_path snapshots (record with preset 'paths')"]
    src = 'odom_ekf' if ctx.sig('odom_ekf') is not None else ctx.pose_ref()[0]
    ref = ctx.sig(src) if src else None
    if ref is None:
        return None, ['skipped: no odometry recorded']
    ts, ds = [], []
    times = [s[0] for s in snaps]
    for i, (ta, poly) in enumerate(snaps):
        tb = times[i + 1] if i + 1 < len(snaps) else ref['t'][-1] + 1.0
        msk = (ref['t'] >= ta) & (ref['t'] < tb)
        if not msk.any():
            continue
        d = sl.point_polyline_dist(ref['x'][msk], ref['y'][msk], poly)
        ts.append(ref['t'][msk])
        ds.append(d)
    if not ts:
        return None, ['skipped: no odometry overlapping the path snapshots']
    t = np.concatenate(ts)
    d = np.concatenate(ds)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t - ctx.t0, d, color=sl.COLORS['path'], lw=1.0)
    shade_states(ax, ctx)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('cross-track [m]')
    ax.set_title(f'Distance from {src} to active /vlm_path_odom')
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.out, 'tracking_error.png'))
    plt.close(fig)

    m = {'pose_source': src, 'n_path_snapshots': len(snaps),
         'crosstrack_rmse_m': round(sl.rmse(d), 4),
         'crosstrack_mean_m': round(float(d.mean()), 4),
         'crosstrack_max_m': round(float(d.max()), 4)}
    return m, [f"cross-track to planned path: RMSE {m['crosstrack_rmse_m']} m, "
               f"mean {m['crosstrack_mean_m']} m, max {m['crosstrack_max_m']} m "
               f"({len(snaps)} snapshots)",
               '![tracking error](tracking_error.png)']


def sec_taps(ctx, plt):
    taps = sl.list_taps(ctx.session)
    if not taps:
        return None, ["skipped: no node-internal taps recorded "
                      "(record with --preset core,debug)"]
    metrics, lines = {}, []
    for name in taps:
        recs = sl.load_tap(ctx.session, name)
        if not recs:
            continue
        scalar, other = sl.tap_keys(recs)
        fig = sl.plot_tap_figure(plt, recs, scalar[:9],
                                 f'debug tap: {name}', t0=ctx.t0)
        fig.savefig(os.path.join(ctx.out, f'tap_{name}.png'))
        plt.close(fig)
        m = {'records': len(recs), 'plottable_keys': scalar, 'array_keys': other}
        _, outcomes = sl.tap_series(recs, 'outcome')
        if outcomes:
            counts = {}
            for v in outcomes:
                counts[str(v)] = counts.get(str(v), 0) + 1
            m['outcome_counts'] = counts
            lines.append(f'`{name}` tick outcomes: '
                         + ', '.join(f'{k}:{v}' for k, v in sorted(counts.items())))
        metrics[name] = m
        lines.append(f'`{name}`: {len(recs)} records; keys: '
                     + ', '.join(scalar + other)
                     + ' — dig deeper with `plot_tap.py <session> ' + name + ' <keys>`')
        lines.append(f'![tap {name}](tap_{name}.png)')
    if not metrics:
        return None, ['skipped: tap files empty']
    return metrics, lines


SECTIONS = [
    ('overview', 'Trajectory overview', sec_overview),
    ('drift', 'Odometry drift', sec_drift),
    ('oscillation', 'Oscillation on straights', sec_oscillation),
    ('signs', 'Sign detection & maneuver engagement', sec_signs),
    ('speed', 'Speed setpoint tracking', sec_speed),
    ('tracking', 'Path tracking (cross-track)', sec_tracking),
    ('taps', 'Node-internal debug taps', sec_taps),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session',
                    help="session dir, session name, or 'latest'")
    ap.add_argument('--sessions-root', default=DEFAULT_SESSIONS,
                    help=f'where sessions live (default: {DEFAULT_SESSIONS})')
    ap.add_argument('--only', default='',
                    help='comma-separated subset of: '
                         + ','.join(k for k, _, _ in SECTIONS))
    args = ap.parse_args()

    session = sl.find_session(args.session, args.sessions_root)
    only = {s for s in args.only.split(',') if s}
    unknown = only - {k for k, _, _ in SECTIONS}
    if unknown:
        raise SystemExit(f'Unknown section(s): {sorted(unknown)}')

    plt = sl.apply_style()
    ctx = Ctx(session)
    print(f'Analyzing {session}')

    all_metrics = {}
    report = [f'# Session report — {os.path.basename(session)}', '']
    if ctx.meta:
        report.append(f"- recorded: {ctx.meta.get('started', '?')}  "
                      f"(commit `{ctx.meta.get('git_commit', '?')}`)")
        if ctx.meta.get('notes'):
            report.append(f"- notes: {ctx.meta['notes']}")
        cnt = ctx.meta.get('message_counts')
        if cnt:
            report.append('- messages: '
                          + ', '.join(f'{k}:{v}' for k, v in cnt.items()))
        report.append('')

    for key, title, fn in SECTIONS:
        if only and key not in only:
            continue
        try:
            metrics, lines = fn(ctx, plt)
        except Exception as e:  # a broken signal must not kill the whole report
            metrics, lines = None, [f'ERROR while analyzing: {e!r}']
        report.append(f'## {title}')
        report.append('')
        for ln in lines:
            report.append(('- ' + ln) if not ln.startswith('!') else ln)
        report.append('')
        if metrics is not None:
            all_metrics[key] = metrics
        status = 'ok' if metrics is not None else lines[0]
        print(f'  {key:12s} {status}')

    with open(os.path.join(ctx.out, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(ctx.out, 'report.md'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    print(f'Report:  {os.path.join(ctx.out, "report.md")}')
    print(f'Metrics: {os.path.join(ctx.out, "metrics.json")}')


if __name__ == '__main__':
    main()
