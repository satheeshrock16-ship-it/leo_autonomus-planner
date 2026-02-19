"""Pipeline wrapper for TLE acquisition."""
from data.fetch_tle import fetch_and_store_tles


def run(limit: int = 200):
    return fetch_and_store_tles(limit=limit)
