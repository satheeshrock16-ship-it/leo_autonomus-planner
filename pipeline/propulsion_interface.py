"""Serial communication layer for hardware thrust command dispatch."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from config import SERIAL_PORT, SERIAL_BAUDRATE

try:
    import serial
except Exception:
    serial = None


@dataclass
class ThrustCommand:
    thrust_vector: list[float]
    duration_ms: int
    burn_type: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def send_command(cmd: ThrustCommand, serial_port: str = SERIAL_PORT, baudrate: int = SERIAL_BAUDRATE) -> None:
    payload = cmd.to_json() + "\n"
    if serial is None:
        raise RuntimeError("pyserial not installed; cannot send command to Arduino.")

    with serial.Serial(serial_port, baudrate=baudrate, timeout=2) as ser:
        ser.write(payload.encode("utf-8"))
