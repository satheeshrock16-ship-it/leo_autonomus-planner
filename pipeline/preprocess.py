"""Pipeline preprocessing: TLE -> ECI state conversion."""
from pathlib import Path
from data.tle_to_eci import convert_default_files


def run(sat_tle_file: Path, debris_tle_file: Path):
    return convert_default_files(sat_tle_file, debris_tle_file)
