# Autonomous LEO Collision Avoidance System

Complete modular prototype for autonomous LEO conjunction screening, maneuver planning, visualization, and hardware thrust-vector command output.

## Folder structure

- `data/`: Space-Track ingestion + TLE→ECI conversion
- `models/`: reserved saved model binaries
- `physics/`: CW propagation, Mahalanobis collision probability, maneuver mechanics
- `ai/`: PINN validator, Bayesian uncertainty estimator, LightGBM decision layer
- `training/`: standalone model training/evaluation scripts
- `pipeline/`: autonomous execution stages and serial propulsion interface
- `visualization/`: 3D plots of pre/post maneuver trajectories
- `hardware/arduino/`: Arduino firmware for 3-servo thrust vector simulation

## Mathematical basis (paper-ready summary)

1. **CW Relative Dynamics** (Hill frame linearization):
   - \(\ddot{x} - 2n\dot{y} - 3n^2 x = 0\)
   - \(\ddot{y} + 2n\dot{x} = 0\)
   - \(\ddot{z} + n^2 z = 0\)

2. **Collision risk via Mahalanobis distance**:
   - \(D^2 = r^T P^{-1} r\)
   - \(P_c \approx e^{-\frac{1}{2}D^2}\)

3. **Maneuver and return burn**:
   - Circular speed / vis-viva laws used for \(\Delta v\) estimation
   - Return correction burn restores nominal radius after conjunction pass

## Autonomous flow

```text
Fetch TLE (Space-Track) -> Convert TLE to ECI (SGP4)
-> Propagate relative motion (CW)
-> Compute Pc (Mahalanobis)
-> AI validation + uncertainty + burn/no-burn classifier
-> If burn: compute avoidance Δv, simulate avoidance/return, emit thrust command JSON
-> Send command to Arduino over serial
```

## Propulsion command format

```json
{
  "thrust_vector": [0.0, 0.85, 0.1],
  "duration_ms": 230,
  "burn_type": "tangential"
}
```

## Hardware integration

Python side: `pipeline/propulsion_interface.py` (serial write)

Arduino side: `hardware/arduino/thrust_vector_servo.ino`
- Receives command as CSV fallback (`x,y,z,duration`) from serial
- Maps vector components to 3 servo gimbal angles
- Holds commanded direction for burn duration, then resets to neutral

## Run demo

```bash
python -m pipeline.run_full_pipeline
```

## Optional dependencies

- `sgp4`, `requests`, `numpy`, `matplotlib`, `pyserial`
- `lightgbm` (optional; fallback model is used if unavailable)
