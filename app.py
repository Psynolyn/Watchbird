import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Search
from geopy.geocoders import Nominatim

class app:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="streamlit_map")
        st.set_page_config(layout="wide")
        
        self.m = ""
        self.returned = ""
        
        if "location" not in st.session_state:
            st.session_state.location = self.geolocator.geocode("Nairobi")
        if "zoom" not in st.session_state:
            st.session_state.zoom = 15
        if "message" not in st.session_state:
            st.session_state.message = []
        if "toast_iterations" not in st.session_state:
            st.session_state.toast_iterations = 0
        if "returnn" not in st.session_state:
            st.session_state.returnn = []
    
    def setup(self):
        self.place = st.sidebar.text_input(label="", placeholder="Search for a place", on_change=self.find, key="place_box", label_visibility="hidden")
       
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
        folium.Marker([st.session_state.location.latitude, st.session_state.location.longitude], popup= st.session_state.location.address).add_to(self.m)
        st.session_state.returnn = st_folium(self.m, height=800, width=None)
        '''
        if ((st.session_state.returnn and "zoom" in  st.session_state.returnn)and
            st.session_state.returnn is not None):
            st.session_state.zoom = st.session_state.returnn["zoom"]
        '''
        
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

if __name__ == '__main__':
    app().run()