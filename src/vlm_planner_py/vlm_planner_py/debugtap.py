"""DebugTap — publish node-INTERNAL signals for the debugkit recorder.

Rationale (docs/testing_pipeline.md §"Node-internal signals"): the session
recorder can only see topics. Internal values (intermediate results, state
machine internals, rejected-frame reasons) never reach a topic, so they are
invisible to a recording. A DebugTap turns selected internals into one JSON
std_msgs/String per tick on /debug/<node_name>, which the recorder's `tap`
kind stores as taps/<name>.jsonl, timestamp-synced with every other signal.

Usage in a node:

    from vlm_planner_py.debugtap import DebugTap

    class MyNode(Node):
        def __init__(self):
            ...
            self.tap = DebugTap(self)          # declares param 'debug_tap' (default off)

        def tick(self):
            ...
            self.tap.put(n_points=len(cloud), thresh=thresh, outcome='published')
            self.tap.put(centerline=pts)       # small [[x,y],...] lists are fine
            ...
            self.tap.flush()                   # one message per tick, stamped now

Toggle live, no restart:   ros2 param set /vlm_planner debug_tap true
Record with:               scripts/record_session.sh --preset core,debug

When the param is false, put()/flush() return immediately — safe to leave the
calls in production code. Values must be JSON-serializable (numbers, strings,
bools, None, small lists); anything else is stored as repr().
"""
import json

from std_msgs.msg import String


class DebugTap:
    def __init__(self, node, topic=None, param_name='debug_tap'):
        self._node = node
        self._param = param_name
        if not node.has_parameter(param_name):
            node.declare_parameter(param_name, False)
        self._pub = node.create_publisher(
            String, topic or ('/debug/' + node.get_name()), 10)
        self._data = {}

    @property
    def enabled(self):
        # Read the param each time so `ros2 param set ... debug_tap true`
        # takes effect immediately.
        return bool(self._node.get_parameter(self._param).value)

    def put(self, **kv):
        """Stage key=value pairs for this tick. No-op while disabled."""
        if not self.enabled:
            return
        for k, v in kv.items():
            self._data[k] = self._jsonable(v)

    def flush(self):
        """Publish everything staged since the last flush as ONE message,
        stamped with the node clock (sim time). No-op if disabled or empty."""
        if not self._data:
            return
        if not self.enabled:        # disabled mid-tick: drop staged leftovers
            self._data = {}
            return
        payload = {'t': self._node.get_clock().now().nanoseconds * 1e-9}
        payload.update(self._data)
        self._data = {}
        self._pub.publish(String(data=json.dumps(payload)))

    @staticmethod
    def _jsonable(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            if isinstance(v, (list, tuple)):
                return [DebugTap._jsonable(x) for x in v]
            return repr(v)
