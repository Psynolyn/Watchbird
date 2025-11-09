import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import db_operations

class app:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="streamlit_map")
        st.set_page_config(layout="wide")
        
        self.m = ""
        self.returned = ""
        self.active_devices = db_operations.get_active_devices() 
        self.model_name = ""
        self.selected_devices = []
        
        if "location" not in st.session_state:
            st.session_state.location = self.geolocator.geocode("Nairobi")
        if "zoom" not in st.session_state:
            st.session_state.zoom = 21
        if "message" not in st.session_state:
            st.session_state.message = []
        if "toast_iterations" not in st.session_state:
            st.session_state.toast_iterations = 0

        if "coords" not in st.session_state:
            st.session_state.returnn = []
    
    def start_data_collection(self):
        st.sidebar.write("Collecting data")
        st.sidebar.write("Available data for training: "+str(db_operations.get_no_of_rows(self.selected_devices)))
        
    def add_learnmode_options(self):
        self.model_name = st.sidebar.text_input(label="Add model name", on_change=self.start_data_collection)

    def setup(self):
        self.place = st.sidebar.text_input(label="Find Location", placeholder="Search for a place", on_change=self.find, key="place_box")
        self.selected_devices = st.sidebar.multiselect("Devices", ["All"]+self.active_devices, placeholder="Select Devices")
        modes = ["Monitor", "Train"] if(self.selected_devices != "All") else ["Monitor"]
        mode = st.sidebar.selectbox("Mode", modes)

        if mode == "Learn":
            self.add_learnmode_options()

        if "All" in self.selected_devices:
            self.selected_devices.extend(self.active_devices)


        self.m = folium.Map(
            location=[st.session_state.location.latitude,  st.session_state.location.longitude],  
            zoom_start=st.session_state.zoom,
            tiles=None 
        )


        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI World Imagery",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(self.m)


        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI Reference Layer",
            name="Labels",
            overlay=True,
            control=True
        ).add_to(self.m)


        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI Roads",
            name="Roads",
            overlay=True,
            control=True
        ).add_to(self.m)

        folium.LayerControl().add_to(self.m)
        popup_msg = st.session_state.location.address + "\n\nLatitude: " + str(st.session_state.location.latitude) + ",\nLongitude: " + str(st.session_state.location.longitude)
        folium.Marker([st.session_state.location.latitude, st.session_state.location.longitude], popup=popup_msg ).add_to(self.m)
        

        if st.session_state.message != []:
            st.session_state.toast_iterations += 1
            st.toast(st.session_state.message[1])

            if  st.session_state.toast_iterations >= 3:
                st.session_state.toast_iterations = 0
                st.session_state.message = []

        
    def find(self):
        query = st.session_state["place_box"].strip()
        if query:
            temp_location = self.geolocator.geocode(query)
            if temp_location:
                st.session_state.location = temp_location
                st.session_state.message = ["sucsess", st.session_state.location.address]
            else:
                st.session_state.message = ["error", "Place not found"]
        else:
            pass

    def run(self):
        self.setup()
        result = st_folium(self.m, height=800, width=None)

        if result.get("last_clicked"):
            lat, lng = result["last_clicked"]["lat"], result["last_clicked"]["lng"]
            st.write("Last Click: ", lat, lng)
        
        '''
        if ((result and "zoom" in  result)and
            result is not None):
            st.session_state.zoom = result["zoom"]
            self.m = folium.Map(
                location=[st.session_state.location.latitude,  st.session_state.location.longitude],  
                zoom_start=st.session_state.zoom,
                tiles=None
                )
            result = st_folium(self.m, height=800, width=None)
            print(result)
        '''
        
            

if __name__ == '__main__':
    app().run()