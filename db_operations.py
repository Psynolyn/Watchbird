from supabase import create_client
import streamlit as st
import pandas as pd
    
SUPABASE_URL = st.secrets["SUPABASE_URL"]  
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]  

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
device_table = []
data_table = []
active_devices = []
polyline_points = {}
device_positions = {}
device_name_to_id = {}
device_id_to_settings = {}
device_id_to_model = {}
device_id_to_data = {}
available_models = []

def load_data(): 
    global device_table, data_table, device_name_to_id, device_id_to_settings, device_id_to_data, active_devices, device_positions, polyline_points, available_models, device_id_to_model
    device_table = supabase.table("Devices").select("*").order("Device_id", desc=False).execute().data
    device_name_to_id = {row["Device_name"] : row ["Device_id"] for row in device_table}
    device_id_to_settings = {row["Device_id"]: row for row in device_table}
    active_devices = [row["Device_name"] for row in device_table if row["Status"] == "Active"]
    device_id_to_model = {row["Device_id"]: row["IF_model"] for row in device_table}

    chunk_size = 1000
    offset = 0

    while True:
        chunk = supabase.table("Data").select("*").order("Id", desc=False).range(offset, offset + chunk_size - 1).execute().data
        if len(chunk) <= 0:
            break
        else:
            data_table.extend(chunk)
            offset = len(data_table)
            
    for row in data_table:
        if(row["Device_id"] not in device_id_to_data):
            device_id_to_data[row["Device_id"]] = []

        device_id_to_data[row["Device_id"]].append(row)

        if row["Device_id"] not in device_positions:
            device_positions[row["Device_id"]] = []

        if row["Device_id"] not in polyline_points:
            polyline_points[row["Device_id"]] = []

        device_positions[row["Device_id"]].append([row["Latitude"], row["Longitude"], row["Anomaly"]])
        polyline_points[row["Device_id"]].append([row["Latitude"], row["Longitude"]])

    for row2 in device_table:
        if row2["IF_model"] not in available_models and row2["IF_model"] != "" and row2["IF_model"] != None:
            available_models.append(row2["IF_model"])

def reload_db():
    global device_table, data_table, device_name_to_id, device_id_to_settings, device_id_to_data, active_devices, device_positions, polyline_points, available_models, device_id_to_model
    device_table = []
    data_table = []
    active_devices = []
    polyline_points = {}
    device_positions = {}
    device_name_to_id = {}
    device_id_to_settings = {}
    device_id_to_model = {}
    device_id_to_data = {}
    available_models = []
    load_data()

def get_active_devices():
    return active_devices

def get_device_positions(device:str):
    id = device_name_to_id.get(device)
    return device_positions[id]

def get_device_polyline_points(device:str):
    id = device_name_to_id.get(device)
    return polyline_points[id]

def get_device_data(device:str):
    id = device_name_to_id.get(device)
    return device_id_to_data.get(id, [])

def get_device_settings(device:str):
    id = device_name_to_id.get(device)
    return device_id_to_settings.get(id)

def get_no_of_rows(devices:list):
    length = 0
    for device in devices:
        length += len(get_device_data(device))
    return length

def save_outlier_result(outlier_result: dict):
    supabase.table("Data").upsert(outlier_result, on_conflict="Id").execute()

def save_device_if_model(if_model: dict):
    supabase.table("Devices").upsert(if_model, on_conflict="Device_name").execute()

def get_device_model(device:str):
    id = device_name_to_id.get(device)
    return device_id_to_model.get(id)

def get_device_data_for_inference(device_name: str):
    """
    Get all data for a device from Supabase for inference.
    Returns DataFrame with all necessary columns for anomaly detection.
    
    Args:
        device_name: Name of the device
    
    Returns:
        pd.DataFrame with columns: Id, Timestamp, Device_id, Latitude, 
        Longitude, Altitude, speed_m_s, delta_alt, dt, distance_m, bearing, acceleration
    """
    
    device_id = device_name_to_id.get(device_name)
    
    if device_id is None:
        return None
    
    try:
        # Get data from Supabase in chunks (to handle large datasets)
        all_data = []
        chunk_size = 1000
        offset = 0
        
        while True:
            # Query Supabase for this device
            chunk = (supabase.table("Data")
                    .select("Id, Timestamp, Device_id, Latitude, Longitude, Altitude, "
                           "speed_m_s, delta_alt, dt, distance_m, bearing, acceleration")
                    .eq("Device_id", device_id)
                    .order("Timestamp", desc=False)
                    .range(offset, offset + chunk_size - 1)
                    .execute()
                    .data)
            
            if len(chunk) == 0:
                break
            
            all_data.extend(chunk)
            
            # If we got less than chunk_size, we've reached the end
            if len(chunk) < chunk_size:
                break
                
            offset += chunk_size
        
        if len(all_data) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure proper data types
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Device_id'] = df['Device_id'].astype(int)
        
        return df
        
    except Exception as e:
        print(f"Error fetching device data from Supabase: {e}")
        return None


def get_unprocessed_device_data(device_name: str, last_id: int = 0):
    """
    Get only unprocessed data points for a device (Id > last_id).
    
    Args:
        device_name: Name of the device
        last_id: Last processed record ID
    
    Returns:
        pd.DataFrame with new records
    """
    
    device_id = device_name_to_id.get(device_name)
    
    if device_id is None:
        return None
    
    try:
        # Get only new records (Id > last_id)
        all_data = []
        chunk_size = 1000
        offset = 0
        
        while True:
            chunk = (supabase.table("Data")
                    .select("Id, Timestamp, Device_id, Latitude, Longitude, Altitude, "
                           "speed_m_s, delta_alt, dt, distance_m, bearing, acceleration")
                    .eq("Device_id", device_id)
                    .gt("Id", last_id)  # Only get records with Id > last_id
                    .order("Timestamp", desc=False)
                    .range(offset, offset + chunk_size - 1)
                    .execute()
                    .data)
            
            if len(chunk) == 0:
                break
            
            all_data.extend(chunk)
            
            if len(chunk) < chunk_size:
                break
                
            offset += chunk_size
        
        if len(all_data) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure proper data types
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Device_id'] = df['Device_id'].astype(int)
        
        return df
        
    except Exception as e:
        print(f"Error fetching unprocessed data from Supabase: {e}")
        return None


# Simpler alternative that uses your existing in-memory data
def get_device_data_for_inference_cached(device_name: str):
    """
    Get device data from already loaded cache (faster but not real-time).
    Uses the device_id_to_data dictionary that's already loaded.
    
    Args:
        device_name: Name of the device
    
    Returns:
        pd.DataFrame with device data
    """
    
    device_id = device_name_to_id.get(device_name)
    
    if device_id is None:
        return None
    
    # Get data from the already loaded cache
    device_data = device_id_to_data.get(device_id, [])
    
    if len(device_data) == 0:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(device_data)
    
    # Ensure proper data types
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    if 'Device_id' in df.columns:
        df['Device_id'] = df['Device_id'].astype(int)
    
    return df