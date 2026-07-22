#!/usr/bin/env python3
"""Hold-to-drive keyboard teleop for the Ackermann robotcar.

Publishes geometry_msgs/Twist on /cmd_vel (linear.x = forward speed,
angular.z = yaw rate; the Gazebo AckermannSteering plugin converts the
yaw rate into a steering angle).

Terminals do not emit key-RELEASE events, so "hold to drive" is emulated:
keys are read non-blocking at publish_rate, and each axis is zeroed if no
keypress for that axis has arrived within key_timeout seconds. Holding a key
fires the terminal's auto-repeat, which keeps the axis alive; releasing it
lets the timeout fire and the car coasts to a stop.

Run this INSTEAD of the MPC / pure_pursuit controller — both publish /cmd_vel.

Controls:
    w / s : drive forward / reverse
    a / d : steer left / right
    space : immediate stop (zero everything)
    q     : quit
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = __doc__


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')

        self.max_linear = self.declare_parameter('max_linear', 1.0).value
        self.max_angular = self.declare_parameter('max_angular', 1.0).value
        self.publish_rate = self.declare_parameter('publish_rate_hz', 20.0).value
        self.key_timeout = self.declare_parameter('key_timeout_sec', 0.3).value
        cmd_topic = self.declare_parameter('command_topic', '/cmd_vel').value

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Per-axis time (node clock, seconds) of the most recent live keypress.
        self._linear_dir = 0.0   # +1 forward, -1 reverse
        self._angular_dir = 0.0  # +1 left,    -1 right
        self._last_linear = -1.0e9
        self._last_angular = -1.0e9

        period = 1.0 / float(self.publish_rate)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f'teleop_keyboard publishing to {cmd_topic} | '
            f'max_linear={self.max_linear} m/s, max_angular={self.max_angular} rad/s')
        self.get_logger().info(HELP)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _read_keys(self):
        """Drain all pending characters from stdin (non-blocking)."""
        keys = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.append(sys.stdin.read(1))
        return keys

    def _on_timer(self):
        now = self._now()
        for k in self._read_keys():
            if k == 'w':
                self._linear_dir = 1.0
                self._last_linear = now
            elif k == 's':
                self._linear_dir = -1.0
                self._last_linear = now
            elif k == 'a':
                self._angular_dir = 1.0
                self._last_angular = now
            elif k == 'd':
                self._angular_dir = -1.0
                self._last_angular = now
            elif k == ' ':
                # Immediate stop: expire both axes now.
                self._last_linear = -1.0e9
                self._last_angular = -1.0e9
            elif k == 'q' or k == '\x03':  # q or Ctrl-C
                raise KeyboardInterrupt

        # Auto-zero any axis whose last live keypress is older than the timeout.
        lin = self._linear_dir if (now - self._last_linear) < self.key_timeout else 0.0
        ang = self._angular_dir if (now - self._last_angular) < self.key_timeout else 0.0

        cmd = Twist()
        cmd.linear.x = lin * self.max_linear
        cmd.angular.z = ang * self.max_angular
        self.cmd_pub.publish(cmd)

    def stop_car(self):
        """Publish a single zero Twist so the car does not keep its last command."""
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboard()

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        # cbreak: deliver keys immediately, no echo, but keep Ctrl-C working.
        tty.setcbreak(fd)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        node.stop_car()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
