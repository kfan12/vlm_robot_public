import json
import re

SIGN_LABELS = ("right", "left", "winding", "stop", "none")


def parse_sign_class(response: str, valid_labels=SIGN_LABELS) -> dict | None:
    """Parse a sign-class VLM response -> {'action': label, 'confidence': float|None}.

    Prefer the JSON object ({"sign":..., "confidence":...}); fall back to keyword
    search over the raw text. Returns None on an unknown/empty/garbled answer so the
    caller FAILS SAFE (no invented class). 'none' is a valid label (no sign in view),
    returned as a dict; None means the read itself failed. Mirrors the validated
    offline parser in scripts/probe_signs_world.py."""
    conf = None
    d = parse_vlm_json(response)
    if isinstance(d, dict):
        s = str(d.get('sign', d.get('action', ''))).strip().lower()
        c = d.get('confidence', None)
        if isinstance(c, (int, float)):
            conf = float(c)
        if s in valid_labels:
            return {'action': s, 'confidence': conf}

    low = response.lower()
    if conf is None:
        m = re.search(r"confidence['\"]?\s*[:=]\s*([01](?:\.\d+)?)", low)
        if m:
            conf = float(m.group(1))
    # priority: explicit glyph words > directional words (so "winding"/"stop" win
    # over a stray "right"/"left" mentioned in passing).
    for kw, lab in (("winding", "winding"), ("wavy", "winding"), ("s-shape", "winding"),
                    ("s-curve", "winding"), ("double", "winding"),
                    ("stop", "stop"), ("octagon", "stop"),
                    ("no sign", "none"), ("none", "none"),
                    ("right", "right"), ("left", "left")):
        if kw in low and lab in valid_labels:
            return {'action': lab, 'confidence': conf}
    return None


def parse_vlm_json(response: str) -> dict | None:
    """Extract JSON from VLM response string."""
    try:
        # Try direct parse first
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block in response
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None