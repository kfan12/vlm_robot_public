#!/usr/bin/env python3
"""Plot node-internal debug-tap signals from a recorded session.

Taps are JSONL files under <session>/taps/ produced by recording a `tap` kind
signal (e.g. /debug/vlm_planner -> taps/planner_tap.jsonl, see
vlm_planner_py/debugtap.py). This tool plots any subset of the tapped keys —
useful for ad-hoc digging beyond the automatic report section:

    python3 tools/debugkit/plot_tap.py latest --list
    python3 tools/debugkit/plot_tap.py latest planner_tap
    python3 tools/debugkit/plot_tap.py latest planner_tap side_thresh side_mem_left side_mem_right outcome
    scripts/analyze_session.sh already renders a default view (analysis/tap_<name>.png)

Output: <session>/analysis/tap_<name>[_custom].png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seslib as sl  # noqa: E402

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_SESSIONS = os.path.join(WS_ROOT, 'data', 'sessions')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session', help="session dir, session name, or 'latest'")
    ap.add_argument('tap', nargs='?', help='tap name (file under <session>/taps/)')
    ap.add_argument('keys', nargs='*', help='keys to plot (default: all plottable)')
    ap.add_argument('--list', action='store_true',
                    help='list available taps and their keys, then exit')
    ap.add_argument('--sessions-root', default=DEFAULT_SESSIONS)
    args = ap.parse_args()

    session = sl.find_session(args.session, args.sessions_root)
    taps = sl.list_taps(session)
    if args.list or not args.tap:
        if not taps:
            raise SystemExit(f'No taps recorded in {session} '
                             "(record with --preset debug / a 'tap' signal).")
        for name in taps:
            recs = sl.load_tap(session, name)
            scalar, other = sl.tap_keys(recs)
            print(f'{name}  ({len(recs)} records)')
            print(f'  plottable: {", ".join(scalar) or "-"}')
            if other:
                print(f'  arrays/objects (not plotted): {", ".join(other)}')
        return
    if args.tap not in taps:
        raise SystemExit(f"Tap '{args.tap}' not found (available: {taps})")

    recs = sl.load_tap(session, args.tap)
    scalar, other = sl.tap_keys(recs)
    keys = args.keys or scalar
    bad = [k for k in keys if k not in scalar]
    if bad:
        raise SystemExit(f'Not plottable: {bad} (plottable keys: {scalar}; '
                         f'array keys: {other})')

    plt = sl.apply_style()
    out_dir = os.path.join(session, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    fig = sl.plot_tap_figure(plt, recs, keys, f'debug tap: {args.tap}',
                             t0=recs[0]['t'] if recs else 0.0)
    suffix = '_custom' if args.keys else ''
    out = os.path.join(out_dir, f'tap_{args.tap}{suffix}.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
