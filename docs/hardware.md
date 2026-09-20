# Sorting and hardware boundaries

The prototype routes an entire image/frame to ACCEPT, REJECT, or REVIEW.
MockActuator records/logs the command; it never opens a serial port or moves
anything. SerialActuator raises NotImplementedError. pyserial is not installed.

Before hardware work is explicitly requested, keep this boundary intact.
A later integration needs a documented device protocol, object tracking,
encoder/belt position, actuator travel time, command IDs, acknowledgements,
timeouts, retry policy, emergency-stop behavior, and a physical fail-safe lane.
Validate those independently with hardware-specific interlocks. SQLite history
alone cannot guarantee exactly-once actuator execution.

Camera mode is bounded by max-frames (default 100). Disconnects raise errors and
release the capture handle. This version saves every inspected frame and can
consume substantial disk space. It does not deduplicate products across frames.
