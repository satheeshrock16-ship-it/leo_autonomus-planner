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


def _execution_burn_time_seconds(burn_time_physics: float) -> float:
    burn_time_physics = float(max(float(burn_time_physics), 0.0))
    if DEMO_MODE:
        return float(burn_time_physics * float(EXECUTION_TIME_SCALE))
    return burn_time_physics


def send_servo_command(yaw: float, pitch: float, serial_conn: object) -> None:
    serial_conn.write(f"{yaw:.2f},{pitch:.2f}\n".encode("utf-8"))


def execute_burn(
    yaw: float,
    pitch: float,
    burn_time_physics: float,
    serial_port: str = SERIAL_PORT,
    baudrate: int = SERIAL_BAUDRATE,
) -> tuple[float, bool]:
    burn_time_physics = float(max(float(burn_time_physics), 0.0))
    burn_time_exec = _execution_burn_time_seconds(burn_time_physics)

    print(f"Burn time (physics): {burn_time_physics:.2f} s")
    print(f"Burn time (execution): {burn_time_exec:.2f} s")

    if REAL_HARDWARE_MODE:
        try:
            if serial is None:
                raise RuntimeError("pyserial is not installed.")

            print("Sending thrust vector command to hardware...")
            with serial.Serial(serial_port, baudrate=baudrate, timeout=2) as serial_conn:
                start_time = time.time()
                while time.time() - start_time < burn_time_exec:
                    send_servo_command(yaw, pitch, serial_conn)
                    time.sleep(0.05)

            print("Burn execution complete.")
            return burn_time_exec, True

        except Exception as exc:
            print("Hardware execution error:", str(exc))
            return burn_time_exec, False

    print("Hardware not connected - running in simulation mode.")
    print(f"Simulated thrust vector: yaw={yaw:.2f}, pitch={pitch:.2f}")
    print(f"Simulated burn duration: {burn_time_exec:.2f} s")
    return burn_time_exec, False


def send_propulsion_command(
    delta_v_vector_km_s: np.ndarray,
    burn_time_seconds: float,
    serial_port: str = SERIAL_PORT,
    baudrate: int = SERIAL_BAUDRATE,
) -> PropulsionCommandResult:
    yaw_deg, pitch_deg = delta_v_vector_to_servo_angles(delta_v_vector_km_s)
    burn_time_physics = float(max(float(burn_time_seconds), 0.0))
    burn_time_exec, connected = execute_burn(
        yaw=yaw_deg,
        pitch=pitch_deg,
        burn_time_physics=burn_time_physics,
        serial_port=serial_port,
        baudrate=baudrate,
    )
    command_string = f"{yaw_deg:.2f},{pitch_deg:.2f},{burn_time_exec:.2f}"
    return PropulsionCommandResult(yaw_deg, pitch_deg, burn_time_physics, connected, command_string)
