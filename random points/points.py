import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import math
import pytz

# --- Configuration ---
N_NORMAL = 280       # Number of "normal" data points
N_TRANSITION = 5   # Number of points to fly to the anomaly zone
N_ANOMALY = 15       # Number of "anomaly" data points
N_TOTAL = N_NORMAL + N_TRANSITION + N_ANOMALY
DEVICE_ID = 3

# Set timezone for Nairobi (EAT)
try:
    nairobi_tz = pytz.timezone('Africa/Nairobi')
except pytz.exceptions.UnknownTimeZoneError:
    nairobi_tz = pytz.timezone('Etc/GMT-3')

START_TIME = nairobi_tz.localize(datetime(2025, 11, 20, 8, 0, 0))
TIME_INTERVAL_MINS = 2 # 2 minutes between GPS pings

# --- Location Parameters ---
CENTER_LAT = -1.2921      # Nairobi center lat
CENTER_LON = 36.8219      # Nairobi center lon
KM_TO_DEG = 1 / 111.0     # Approx conversion factor for lat/lon at equator

# Normal Zone (4km radius)
NORMAL_RADIUS_KM = 4.0
NORMAL_RADIUS_DEG = NORMAL_RADIUS_KM * KM_TO_DEG

# Anomaly Zone (Now 300m / 0.3km *away* from the edge of the 4km zone)
# Total distance from center = 4km (normal) + 0.3km (new distance) = 4.3km
ANOMALY_DISTANCE_KM = 4.3 
ANOMALY_RADIUS_KM = 0.1   # A small 0.1km (100m) "infested area"
ANOMALY_RADIUS_DEG = ANOMALY_RADIUS_KM * KM_TO_DEG

# Place anomaly zone 4.3km North-East of the center
ANOMALY_CENTER_LAT = CENTER_LAT + (ANOMALY_DISTANCE_KM * KM_TO_DEG * 0.707)
ANOMALY_CENTER_LON = CENTER_LON + (ANOMALY_DISTANCE_KM * KM_TO_DEG * 0.707)

# --- Altitude Parameters ---
NORMAL_ALT_BASE = 1800.0  # Normal flight altitude (e.g., 1800-1900m)
NORMAL_ALT_RANGE = 100.0
ANOMALY_ALT_BASE = 1750.0 # Lower altitude for feeding (e.g., 1750-1770m)
ANOMALY_ALT_RANGE = 20.0

# --- Data Generation ---
lats = []
lons = []
alts = []
times = []
anomalies = []

# Initialize starting position inside the normal zone
current_lat = CENTER_LAT + random.uniform(-NORMAL_RADIUS_DEG, NORMAL_RADIUS_DEG)
current_lon = CENTER_LON + random.uniform(-NORMAL_RADIUS_DEG, NORMAL_RADIUS_DEG)
current_time = START_TIME

print(f"Generating {N_NORMAL} normal points...")
# Loop 1: Normal Zone (Random walk within 4km radius)
for i in range(N_NORMAL):
    # Small random step
    step_lat = random.uniform(-0.002, 0.002)
    step_lon = random.uniform(-0.002, 0.002)
    
    # Check bounds: if a step takes it outside, reverse the step and try a new one
    next_lat = current_lat + step_lat
    next_lon = current_lon + step_lon
    distance_from_center = math.sqrt((next_lat - CENTER_LAT)**2 + (next_lon - CENTER_LON)**2)
    
    if distance_from_center > NORMAL_RADIUS_DEG:
        # If outside, reverse step (bounce back)
        current_lat -= step_lat
        current_lon -= step_lon
    else:
        # If inside, accept step
        current_lat = next_lat
        current_lon = next_lon
        
    alt = NORMAL_ALT_BASE + random.uniform(0, NORMAL_ALT_RANGE)
    
    lats.append(current_lat)
    lons.append(current_lon)
    alts.append(alt)
    times.append(current_time.strftime('%Y-%m-%d %H:%M:%S%z'))
    anomalies.append(1) # 1 = Normal
    
    current_time += timedelta(minutes=TIME_INTERVAL_MINS)

print(f"Generating {N_TRANSITION} transition points...")
# Loop 2: Transition Zone (Fly from last normal point to anomaly zone)
start_fly_lat = lats[-1]
start_fly_lon = lons[-1]

for i in range(N_TRANSITION):
    # Linear interpolation + a bit of noise
    fraction = (i + 1) / float(N_TRANSITION)
    noise_lat = random.uniform(-0.0005, 0.0005)
    noise_lon = random.uniform(-0.0005, 0.0005)

    current_lat = start_fly_lat + (ANOMALY_CENTER_LAT - start_fly_lat) * fraction + noise_lat
    current_lon = start_fly_lon + (ANOMALY_CENTER_LON - start_fly_lon) * fraction + noise_lon
    
    # Altitude can drop during transition
    alt_fraction = (NORMAL_ALT_BASE + NORMAL_ALT_RANGE) - (NORMAL_ALT_BASE - ANOMALY_ALT_BASE) * fraction
    alt = alt_fraction + random.uniform(0, ANOMALY_ALT_RANGE)
    
    lats.append(current_lat)
    lons.append(current_lon)
    alts.append(alt)
    times.append(current_time.strftime('%Y-%m-%d %H:%M:%S%z'))
    anomalies.append(1) # Still 1 (normal), just flying purposefully
    
    current_time += timedelta(minutes=TIME_INTERVAL_MINS)

print(f"Generating {N_ANOMALY} anomaly points...")
# Loop 3: Anomaly Zone (Random walk within the 0.1km "pest area")
for i in range(N_ANOMALY):
    # Small random step
    step_lat = random.uniform(-0.0002, 0.0002)
    step_lon = random.uniform(-0.0002, 0.0002)
    
    next_lat = current_lat + step_lat
    next_lon = current_lon + step_lon
    
    # Bounds check for the *anomaly* zone
    distance_from_anomaly_center = math.sqrt((next_lat - ANOMALY_CENTER_LAT)**2 + (next_lon - ANOMALY_CENTER_LON)**2)
    
    if distance_from_anomaly_center > ANOMALY_RADIUS_DEG:
        # Bounce back
        current_lat -= step_lat
        current_lon -= step_lon
    else:
        current_lat = next_lat
        current_lon = next_lon
        
    alt = ANOMALY_ALT_BASE + random.uniform(0, ANOMALY_ALT_RANGE)
    
    lats.append(current_lat)
    lons.append(current_lon)
    alts.append(alt)
    times.append(current_time.strftime('%Y-%m-%d %H:%M:%S%z'))
    anomalies.append(-1) # -1 = Anomaly
    
    current_time += timedelta(minutes=TIME_INTERVAL_MINS)

print(f"Total points generated: {len(lats)}")

# --- SQL File Generation ---
sql_statements = []
# Use case-sensitive column names as requested
sql_statements.append('INSERT INTO "Data" ("Timestamp", "Latitude", "Longitude", "Altitude", "Device_id", "Anomaly") VALUES')

values = []
for i in range(N_TOTAL):
    # Format: ('timestamp', lat, lon, alt, dev_id, anomaly)
    val = f"('{times[i]}', {lats[i]:.6f}, {lons[i]:.6f}, {alts[i]:.2f}, {DEVICE_ID}, {anomalies[i]})"
    values.append(val)

# Join all values with commas, and end the statement with a semicolon
sql_statements.append(",\n".join(values) + ";")

# Write to file
file_name = 'bird_data_300m_dist.sql'
with open(file_name, 'w') as f:
    f.write("\n".join(sql_statements))

print(f"Successfully generated SQL script: {file_name}")