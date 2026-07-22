"""Shared helpers for the debugkit analysis tools: session loading, signal
math (interpolation, segments, spectra, polyline distance) and a consistent
matplotlib style. No ROS dependency — works on any saved session folder."""
import csv
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Plot style — fixed entity->color mapping so every figure reads the same.
# Palette values are the dataviz reference categorical set (light mode).
# ---------------------------------------------------------------------------

COLORS = {
    'odom_truth': '#52514e',   # ground truth = neutral dark reference
    'odom':       '#2a78d6',   # wheel odometry = blue
    'odom_ekf':   '#1baf7a',   # EKF estimate = aqua
    'cmd':        '#eda100',   # commanded (cmd_vel, target speed) = yellow
    'actual':     '#2a78d6',   # measured response = blue
    'sign':       '#e34948',   # VLM sign events = red
    'pending':    '#1f9e8f',   # arbitrated candidate maneuver (pending_sign) = teal
    'path':       '#4a3aa7',   # planner path snapshots = violet
    'imu':        '#e87ba4',   # imu-derived = magenta
    'accent':     '#eb6834',   # segment highlights = orange
}
STATE_COLORS = {               # maneuver state bands (light tints of the above)
    'straight': '#ffffff',
    'left':     '#d8f3e8',
    'right':    '#d4e4f7',
    'winding':  '#fdeccc',
    'stop':     '#f9dada',
    'none':     '#ffffff',
}
GRID = '#e1e0d9'
MUTED = '#898781'
INK = '#0b0b0b'
INK2 = '#52514e'
SURFACE = '#fcfcfb'


def apply_style():
    import matplotlib
    if 'ipykernel' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelcolor': INK2,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'lines.linewidth': 1.5,
        'legend.frameon': False,
        'figure.dpi': 110,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
    return plt


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def load_table(path):
    """CSV -> dict of numpy arrays. Numeric columns become float64; columns
    with any non-numeric entry stay as object (string) arrays. None if the
    file is missing or has no data rows."""
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        rows = [r for r in reader if r]
    if not rows:
        return None
    cols = {}
    for i, name in enumerate(header):
        raw = [r[i] if i < len(r) else '' for r in rows]
        try:
            cols[name] = np.array([float(v) for v in raw])
        except ValueError:
            cols[name] = np.array(raw, dtype=object)
    return cols


def load_signal(session, name):
    """signals/<name>.csv -> column dict (or None)."""
    return load_table(os.path.join(session, 'signals', name + '.csv'))


def sign_label(v):
    """Bare label from a /vlm/sign value: plain string or JSON payload."""
    s = str(v).strip()
     # Debugging line
    if s.startswith('{'):
        import json
        try:
            return str(json.loads(s).get('sign', s))
        except (ValueError, AttributeError):
            return s
    return s


def load_path_snapshots(session, name, max_snapshots=0):
    """paths/<name>/ -> list of (t, Nx2 array), time-sorted. [] if absent."""
    d = os.path.join(session, 'paths', name)
    idx = load_table(os.path.join(d, 'index.csv'))
    if idx is None:
        return []
    # The planner nodes stamp Path headers with the WALL clock (they do not
    # run under use_sim_time), while every other signal is on the sim epoch.
    # If t disagrees with the recorder's sim-clock t_recv by more than an
    # hour, trust t_recv so tracking analyses can align paths with odometry.
    if 't_recv' in idx and len(idx['t']) and \
            float(np.median(np.abs(idx['t'] - idx['t_recv']))) > 3600.0:
        idx['t'] = idx['t_recv']
    order = np.argsort(idx['t'])
    if max_snapshots and len(order) > max_snapshots:
        step = len(order) / max_snapshots
        order = [order[int(i * step)] for i in range(max_snapshots)]
    snaps = []
    for i in order:
        tab = load_table(os.path.join(d, str(idx['file'][i])))
        if tab is None or 'x' not in tab or len(tab['x']) < 2:
            continue
        snaps.append((float(idx['t'][i]), np.column_stack([tab['x'], tab['y']])))
    return snaps


def load_image_index(session, name):
    return load_table(os.path.join(session, 'images', name, 'index.csv'))


def load_tap(session, name):
    """taps/<name>.jsonl -> list of dict records, time-sorted. [] if absent."""
    import json
    p = os.path.join(session, 'taps', name + '.jsonl')
    if not os.path.isfile(p):
        return []
    recs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and 't' in r:
                recs.append(r)
    recs.sort(key=lambda r: r['t'])
    return recs


def list_taps(session):
    d = os.path.join(session, 'taps')
    if not os.path.isdir(d):
        return []
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith('.jsonl'))


def tap_keys(recs):
    """Split a tap's keys into (scalar, other). Scalars are plottable by
    plot_tap_figure: numbers, bools, and short strings (categorical)."""
    scalar, other = [], []
    for r in recs:
        for k, v in r.items():
            if k in ('t', 't_recv'):
                continue
            if isinstance(v, (int, float, bool)) or \
                    (isinstance(v, str) and len(v) <= 40):
                if k not in scalar and k not in other:
                    scalar.append(k)
            elif k not in other:
                if k in scalar:
                    scalar.remove(k)
                other.append(k)
    return scalar, other


def tap_series(recs, key):
    """(t_array, value_list) of the records where `key` is present, non-None."""
    pts = [(r['t'], r[key]) for r in recs if r.get(key) is not None]
    return (np.array([p[0] for p in pts]), [p[1] for p in pts])


def plot_tap_figure(plt, recs, keys, title, t0=0.0):
    """One stacked subplot per key: numeric -> step line; bool -> 0/1 step;
    string -> categorical step. Returns the figure (caller saves/closes)."""
    keys = [k for k in keys if len(tap_series(recs, k)[0])]
    n = max(len(keys), 1)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.5 * n + 0.8), sharex=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, key in zip(axes, keys):
        t, v = tap_series(recs, key)
        if all(isinstance(x, bool) for x in v):
            ax.step(t - t0, [int(x) for x in v], where='post',
                    color=COLORS['accent'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['false', 'true'])
        elif all(isinstance(x, (int, float)) for x in v):
            ax.step(t - t0, [float(x) for x in v], where='post',
                    color=COLORS['actual'], lw=1.2)
        else:
            labels = sorted({str(x) for x in v})
            ymap = {l: i for i, l in enumerate(labels)}
            ax.step(t - t0, [ymap[str(x)] for x in v], where='post',
                    color=COLORS['path'])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylabel(key, fontsize=8)
    if not keys:
        axes[0].text(0.5, 0.5, 'no plottable keys', ha='center',
                     transform=axes[0].transAxes, color=MUTED)
    axes[0].set_title(title)
    axes[-1].set_xlabel('t [s]')
    fig.tight_layout()
    return fig


def load_meta(session):
    import yaml
    p = os.path.join(session, 'meta.yaml')
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_world_signs(world_path):
    """Parse a Gazebo world SDF for `<include>` blocks named `sign_<label>`
    and return {label: (x, y)} in world-frame coordinates (from the model's
    <pose>). Used to overlay the true sign-board positions on the trajectory
    plot; returns {} if the file is missing or has no sign includes, rather
    than raising, since this is a display extra, not a required signal."""
    if not world_path or not os.path.isfile(world_path):
        return {}
    import re
    import xml.etree.ElementTree as ET
    with open(world_path, encoding='utf-8') as f:
        text = f.read()
    # Some auto-generated worlds embed a bare "--" inside a <!-- --> comment
    # (e.g. "-- safe by construction --"), which is invalid per the XML spec
    # even though Gazebo's own (lenient) parser accepts it. Strip comments
    # rather than fail on them -- they carry no data we need here.
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    out = {}
    for inc in root.iter('include'):
        name_el = inc.find('name')
        pose_el = inc.find('pose')
        if name_el is None or pose_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        if not name.startswith('sign_'):
            continue
        label = name[len('sign_'):]
        parts = pose_el.text.split()
        if len(parts) < 2:
            continue
        try:
            out[label] = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue
    return out


def find_session(arg, sessions_root):
    """Resolve a session argument: an existing dir, a name under the sessions
    root, or the literal 'latest'."""
    if os.path.isdir(arg) and arg != 'latest':
        return os.path.abspath(arg)
    if arg == 'latest':
        if not os.path.isdir(sessions_root):
            raise SystemExit(f'No sessions root at {sessions_root}')
        subdirs = sorted(d for d in os.listdir(sessions_root)
                         if os.path.isdir(os.path.join(sessions_root, d)))
        if not subdirs:
            raise SystemExit(f'No sessions found under {sessions_root}')
        return os.path.join(sessions_root, subdirs[-1])
    cand = os.path.join(sessions_root, arg)
    if os.path.isdir(cand):
        return cand
    raise SystemExit(f"Session '{arg}' not found (looked in {sessions_root})")


# ---------------------------------------------------------------------------
# Signal math
# ---------------------------------------------------------------------------

def interp(tq, t, v):
    """Linear interp of v(t) at query times tq, clamped at the ends."""
    return np.interp(tq, t, v)


def step_interp(tq, t, v):
    """Zero-order hold (previous value). Before the first sample -> v[0]."""
    idx = np.searchsorted(t, tq, side='right') - 1
    idx = np.clip(idx, 0, len(v) - 1)
    return np.asarray(v)[idx]


def unwrap(a):
    return np.unwrap(np.asarray(a, dtype=float))


def ang_diff(a, b):
    """Smallest signed angle a-b."""
    d = np.asarray(a) - np.asarray(b)
    return (d + np.pi) % (2 * np.pi) - np.pi


def cumdist(x, y):
    d = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(d)])


def dedup_events(t, labels):
    """Keep only samples where the label changes (first sample always kept)."""
    t = np.asarray(t)
    labels = np.asarray(labels, dtype=object)
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = labels[1:] != labels[:-1]
    return t[keep], labels[keep]


def label_segments(t, labels, t_end=None):
    """Change events -> [(label, t_start, t_stop), ...] covering the record."""
    te, le = dedup_events(t, labels)
    if len(te) == 0:
        return []
    stops = np.concatenate([te[1:], [t_end if t_end is not None else t[-1]]])
    return [(str(le[i]), float(te[i]), float(stops[i])) for i in range(len(te))]


def segments_where(t, labels, target, min_duration=0.0, t_end=None):
    return [(a, b) for lab, a, b in label_segments(t, labels, t_end)
            if lab == target and (b - a) >= min_duration]


def line_fit_lateral(x, y):
    """Fit the principal line through (x,y) (PCA) and return the signed
    perpendicular deviation of each point from it, plus the along-track
    coordinate. Robust way to measure wobble on a nominally straight run."""
    pts = np.column_stack([x, y]).astype(float)
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q / max(len(pts) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]          # unit principal direction
    if axis @ (pts[-1] - pts[0]) < 0:
        axis = -axis                           # point along travel
    along = q @ axis
    lateral = q[:, 0] * (-axis[1]) + q[:, 1] * axis[0]   # left-positive
    return along, lateral


def dominant_freq(t, v, f_min=0.05):
    """Dominant frequency (Hz) of v(t) plus the one-sided amplitude spectrum.
    Resamples onto a uniform grid at the median sample interval first.
    Returns (freq, freqs, amp); freq is None if the signal is too short."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 16 or t[-1] - t[0] <= 0:
        return None, None, None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None, None, None
    tu = np.arange(t[0], t[-1], dt)
    vu = np.interp(tu, t, v)
    vu = vu - vu.mean()
    n = len(vu)
    if n < 16:
        return None, None, None
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(vu * win)) * 2.0 / win.sum()
    freqs = np.fft.rfftfreq(n, dt)
    mask = freqs >= f_min
    if not mask.any() or amp[mask].max() <= 0:
        return None, freqs, amp
    return float(freqs[mask][np.argmax(amp[mask])]), freqs, amp


def point_polyline_dist(px, py, poly):
    """Min distance from each point (px,py) to the polyline (Mx2).
    Vectorized over segments per point batch."""
    p = np.column_stack([np.atleast_1d(px), np.atleast_1d(py)]).astype(float)
    a = poly[:-1]                    # (S,2) segment starts
    b = poly[1:]                     # (S,2) segment ends
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1)
    ab2[ab2 == 0] = 1e-12
    # (N,S) projection parameter, clamped to the segment
    ap = p[:, None, :] - a[None, :, :]
    tpar = np.clip((ap * ab[None, :, :]).sum(axis=2) / ab2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + tpar[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(p[:, None, :] - closest, axis=2)
    return d.min(axis=1)


def rmse(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')
