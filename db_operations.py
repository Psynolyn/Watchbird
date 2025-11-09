from supabase import create_client
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]  
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]  

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_active_devices():
    response = supabase.table("Devices").select("*").execute().data
    return [row["Device_name"] for row in response if row["Status"] == "Active"]

def get_no_of_rows(devices:list):
    response = supabase.table("Devices").select("Device_id, Device_name").in_("Device_name", devices).execute().data
    ids = [row["Device_id"] for row in response]
    return supabase.table("Data").select("*", count="exact").in_("Device_id", ids).execute().count

print()   
