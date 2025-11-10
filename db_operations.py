from supabase import create_client
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]  
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]  

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
device_table = []
data_table = []
active_devices = []
device_name_to_id = {}
device_id_to_settings = {}
device_id_to_data = {}

device_table = supabase.table("Devices").select("*").order("Device_id", desc=False).execute().data
data_table = supabase.table("Data").select("*").order("Id", desc=False).execute().data

def load_data():
    global device_table, data_table, device_name_to_id, device_id_to_settings, device_id_to_data, active_devices
    device_table = supabase.table("Devices").select("*").order("Device_id", desc=False).execute().data
    data_table = supabase.table("Data").select("*").order("Id", desc=False).execute().data
    device_name_to_id = {row["Device_name"] : row ["Device_id"] for row in device_table}
    device_id_to_settings = {row["Device_id"]: row for row in device_table}
    active_devices = [row["Device_name"] for row in device_table if row["Status"] == "Active"]

    for row in data_table:
        if(row["Device_id"] not in device_id_to_data):
            device_id_to_data[row["Device_id"]] = []
            device_id_to_data[row["Device_id"]].append(row)
        else:
            device_id_to_data[row["Device_id"]].append(row)

def get_active_devices():
    return active_devices

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