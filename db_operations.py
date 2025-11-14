from supabase import create_client
import streamlit as st

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
