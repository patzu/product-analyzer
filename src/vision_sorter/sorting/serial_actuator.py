"""Hardware integration intentionally disabled until explicitly designed."""


class SerialActuator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Real hardware is not enabled. Use MockActuator; see docs/hardware.md.")
