"""Two-servo propulsion command interface with safe simulation fallback."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from config import SERIAL_BAUDRATE, SERIAL_PORT

try:
    import serial
except Exception:  # pragma: no cover - optional dependency
    serial = None


@dataclass
class PropulsionCommandResult:
    yaw_deg: float
    pitch_deg: float
    burn_time_seconds: float
    connected: bool
    command_string: str


def delta_v_vector_to_servo_angles(delta_v_vector_km_s: np.ndarray) -> tuple[float, float]:
    dv = np.asarray(delta_v_vector_km_s, dtype=float)
    x, y, z = float(dv[0]), float(dv[1]), float(dv[2])
    azimuth_rad = math.atan2(y, x)  # [-pi, pi]
    xy_norm = math.sqrt(x * x + y * y)
    elevation_rad = math.atan2(z, max(xy_norm, 1e-12))  # [-pi/2, pi/2]

    yaw_deg = float(np.clip(math.degrees(azimuth_rad) + 90.0, 0.0, 180.0))
    pitch_deg = float(np.clip(math.degrees(elevation_rad) + 90.0, 0.0, 180.0))
    return yaw_deg, pitch_deg


def send_propulsion_command(
    delta_v_vector_km_s: np.ndarray,
    burn_time_seconds: float,
    serial_port: str = SERIAL_PORT,
    baudrate: int = SERIAL_BAUDRATE,
) -> PropulsionCommandResult:
    yaw_deg, pitch_deg = delta_v_vector_to_servo_angles(delta_v_vector_km_s)
    burn_time_seconds = float(max(float(burn_time_seconds), 0.0))
    command_string = f"{yaw_deg:.2f},{pitch_deg:.2f},{burn_time_seconds:.2f}"

    if serial is None:
        print("Hardware not connected – running in simulation mode.")
        print("Simulated signal data:")
        print("  Yaw (deg):", f"{yaw_deg:.2f}")
        print("  Pitch (deg):", f"{pitch_deg:.2f}")
        print("  Burn Time (s):", f"{burn_time_seconds:.2f}")
        print("  Serial Command:", command_string)
        return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time_seconds, False, command_string)

    try:
        with serial.Serial(serial_port, baudrate=baudrate, timeout=2) as ser:
            ser.write((command_string + "\n").encode("utf-8"))
        print("Hardware connected.")
        print("Signal sent:", command_string)
        connected = True
    except Exception:
        print("Hardware not connected – running in simulation mode.")
        print("Simulated signal data:")
        print("  Yaw (deg):", f"{yaw_deg:.2f}")
        print("  Pitch (deg):", f"{pitch_deg:.2f}")
        print("  Burn Time (s):", f"{burn_time_seconds:.2f}")
        print("  Serial Command:", command_string)
        connected = False

    return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time_seconds, connected, command_string)
