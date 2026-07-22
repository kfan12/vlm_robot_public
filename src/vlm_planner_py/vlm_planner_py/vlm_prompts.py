# Constrained, no-leak sign classifier (validated offline in
# scripts/probe_signs_world.py, Phase 1b). Each label is grounded by the SYMBOL on
# the sign face (not by which way to drive), so the phrasing cannot bias the answer
# toward a direction (vlm_integration_plan R2 "prompt leakage" lesson). Consumed via
# vlm_node._run_vlm -> json_parser.parse_sign_class.
SIGN_PROMPT  = (
    "You are the perception system of a small car with a front camera, driving on a "
    "dark road marked by two white lines. A single road sign may be standing at the "
    "right edge of the road ahead.\n"
    "Classify the sign by the SYMBOL on its face into exactly ONE of these labels:\n"
    '  "right"   - a black arrow that bends toward the right\n'
    '  "left"    - a black arrow that bends toward the left\n'
    '  "winding" - a black wavy, S-shaped (double-bend) arrow\n'
    '  "stop"    - white letters STOP on a red octagon\n'
    '  "none"    - no sign is clearly visible or readable\n'
    "If several signs are visible, classify only the LARGEST (closest) one.\n"
    'Reply with ONE line of JSON and nothing else: '
    '{"sign": "<right|left|winding|stop|none>"}'
)


DIRECTION_PROMPT = """Look at this image from a car's front camera.
{task_description}
Based on what you see, what is the best move? Answer with exactly one word:
left, center, right, stop. Output only the word.
"""

SEMANTIC_PROMPT = """You are the navigation system of a small car with a front camera.

Task: {task_description}

Output ONE line of JSON with exactly these keys and real values for the current image:
- target_visible: true or false (is the task's target visible?)
- confidence: a number between 0.0 and 1.0
- driving_direction: one of left, center, right, stop (use stop only if no target or blocked)
- target_description: a few words for what you see ahead
- reason: a few words

Here is an example answer for a DIFFERENT image (do not copy these values):
{{"target_visible": true, "confidence": 0.7, "driving_direction": "center", "target_description": "two yellow cones with a gap ahead", "reason": "drive through the gap"}}

Now give the JSON for THIS image. Output only the JSON line, nothing else.
"""

WAYPOINT_PROMPT = """You are a robot navigation planner.

Task: {task_description}

Image size: {width}x{height} pixels.

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{{
  "target_visible": true,
  "confidence": 0.0,
  "waypoints_px": [[u1, v1], [u2, v2], [u3, v3]],
  "driving_direction": "left|center|right|stop",
  "reason": "short explanation"
}}

Rules:
- waypoints_px: 8 to 12 pixel coordinate pairs [u, v] where u=column, v=row
- First waypoint near bottom center of image (near robot)
- Waypoints progress forward (decreasing v = further away)
- u range: 0 to {width}, v range: 0 to {height}
- If target not visible, set driving_direction to "stop" and waypoints_px to []
"""