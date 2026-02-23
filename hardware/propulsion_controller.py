"""Two-servo propulsion command interface with safe simulation fallback."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from config import DEMO_MODE, EXECUTION_TIME_SCALE, REAL_HARDWARE_MODE, SERIAL_BAUDRATE, SERIAL_PORT

try:
    import serial
except Exception:  # pragma: no cover - optional dependency
    serial = None

SIMULATION_MODE_MESSAGE = "Hardware not connected \u2013 running in simulation mode."


@dataclass
class PropulsionCommandResult:
    yaw_deg: float
    pitch_deg: float
    burn_time_seconds: float
    connected: bool
    command_string: str


def delta_v_vector_to_servo_angles(delta_v_vector_km_s: np.ndarray) -> tuple[float, float]:
    dv = np.asarray(delta_v_vector_km_s, dtype=float).reshape(3)
    x, y, z = float(dv[0]), float(dv[1]), float(dv[2])
    azimuth_rad = math.atan2(y, x)  # [-pi, pi]
    xy_norm = math.sqrt(x * x + y * y)
    elevation_rad = math.atan2(z, max(xy_norm, 1e-12))  # [-pi/2, pi/2]

    yaw_deg = float(np.clip(math.degrees(azimuth_rad) + 90.0, 0.0, 180.0))
    pitch_deg = float(np.clip(math.degrees(elevation_rad) + 90.0, 0.0, 180.0))
    return yaw_deg, pitch_deg


def _execution_burn_time_seconds(burn_time_seconds: float) -> float:
    burn_time_seconds = float(max(float(burn_time_seconds), 0.0))
    if DEMO_MODE:
        scaled = burn_time_seconds * float(EXECUTION_TIME_SCALE)
        return float(np.clip(scaled, 0.0, burn_time_seconds))
    return burn_time_seconds


def _format_command_string(yaw_deg: float, pitch_deg: float, burn_time_seconds: float) -> str:
    yaw = float(np.clip(float(yaw_deg), 0.0, 180.0))
    pitch = float(np.clip(float(pitch_deg), 0.0, 180.0))
    burn_time = float(max(float(burn_time_seconds), 0.0))
    return f"{yaw:.2f},{pitch:.2f},{burn_time:.2f}"


def _send_serial_signal(serial_conn: object, signal: str) -> None:
    serial_conn.write(f"{signal}\n".encode("utf-8"))
    serial_conn.flush()


def execute_burn(
    delta_v_vector: np.ndarray,
    burn_time_seconds: float,
    serial_port: str = SERIAL_PORT,
    baudrate: int = SERIAL_BAUDRATE,
) -> PropulsionCommandResult:
    burn_time = float(max(float(burn_time_seconds), 0.0))
    burn_time_exec = _execution_burn_time_seconds(burn_time)

    try:
        yaw_deg, pitch_deg = delta_v_vector_to_servo_angles(delta_v_vector)
    except Exception:
        yaw_deg, pitch_deg = 90.0, 90.0
    yaw_deg = float(np.clip(yaw_deg, 0.0, 180.0))
    pitch_deg = float(np.clip(pitch_deg, 0.0, 180.0))
    command_string = _format_command_string(yaw_deg, pitch_deg, burn_time)

    if REAL_HARDWARE_MODE:
        try:
            if serial is None:
                raise RuntimeError("pyserial is not installed.")
            with serial.Serial(serial_port, baudrate=baudrate, timeout=2) as serial_conn:
                _send_serial_signal(serial_conn, command_string)
                time.sleep(burn_time_exec)
                # Immediately command neutral orientation at burn completion.
                _send_serial_signal(serial_conn, "90.00,90.00,0.00")
            return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time, True, command_string)
        except Exception:
            print(SIMULATION_MODE_MESSAGE)
            print(f"Simulated signal: {command_string}")
            time.sleep(burn_time_exec)
            return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time, False, command_string)

    print(SIMULATION_MODE_MESSAGE)
    print(f"Simulated signal: {command_string}")
    time.sleep(burn_time_exec)
    return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time, False, command_string)


def send_propulsion_command(
    delta_v_vector_km_s: np.ndarray,
    burn_time_seconds: float,
    serial_port: str = SERIAL_PORT,
    baudrate: int = SERIAL_BAUDRATE,
) -> PropulsionCommandResult:
    return execute_burn(
        delta_v_vector=delta_v_vector_km_s,
        burn_time_seconds=burn_time_seconds,
        serial_port=serial_port,
        baudrate=baudrate,
    )
