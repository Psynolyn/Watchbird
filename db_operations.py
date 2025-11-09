from supabase import create_client
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]  
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]  

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
device_table = []
data_table = []

def load_data():
    global device_table, data_table
    device_table = supabase.table("Devices").select("*").order("Device_id", desc=False).execute().data
    data_table = supabase.table("Data").select("*").order("Id", desc=False).execute().data

def get_active_devices():
    return [row["Device_name"] for row in device_table if row["Status"] == "Active"]

def get_device_data(device:str):
    id = [row["Device_id"] for row in device_table if row["Device_name"] == device][0]
    return [row for row in data_table if row['Device_id'] == id]

def get_device_settings(device:str):
    id = [row["Device_id"] for row in device_table if row["Device_name"] == device][0]
    return device_table[id-1]

def get_no_of_rows(devices:list):
    ids = [row["Device_id"] for row in device_table if row["Device_name"] in devices]
    return len([row["Device_id"] for row in data_table if row["Device_id"] in ids])
