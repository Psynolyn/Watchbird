import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
table_name = "Data"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return np.degrees(np.arctan2(x, y))

def fetch_data(table_name: str):
    response = supabase.table(table_name).select("*").execute()
    data = response.data
    df = pd.DataFrame(data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

def compute_features(df: pd.DataFrame):
    df = df.sort_values(["Device_id", "Timestamp"]).reset_index(drop=True)
    
    df["dt"] = df["Timestamp"].diff().dt.total_seconds().fillna(0)
    df["delta_lat"] = df["Latitude"].diff().fillna(0)
    df["delta_lon"] = df["Longitude"].diff().fillna(0)
    df["delta_alt"] = df["Altitude"].diff().fillna(0)

    df["distance_m"] = haversine(df["Latitude"].shift(), df["Longitude"].shift(),
                                 df["Latitude"], df["Longitude"]).fillna(0)
    
    df["speed_m_s"] = df["distance_m"] / df["dt"].replace(0, np.nan)
    df["speed_m_s"] = df["speed_m_s"].fillna(0)
    
    df["bearing"] = bearing(df["Latitude"].shift(), df["Longitude"].shift(),
                            df["Latitude"], df["Longitude"]).fillna(0)
    
    df["acceleration"] = df["speed_m_s"].diff().fillna(0) / df["dt"].replace(0, np.nan)
    df["acceleration"] = df["acceleration"].fillna(0)
    
    df["cumulative_distance"] = df.groupby("Device_id")["distance_m"].cumsum()
    
    return df

def prepare_dataset():
    df = fetch_data("Data")
    df_features = compute_features(df)
    df_features.to_csv("dataset\derived.csv", index=False)
