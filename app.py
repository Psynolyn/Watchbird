import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import db_operations

class app:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="streamlit_map")
        st.set_page_config(layout="wide")
 
        self.returned = ""
        self.active_devices = db_operations.get_active_devices() 
        st.session_state.build_mapodel_name = ""
        self.selected_devices = []
        
        
        if "location" not in st.session_state:
            st.session_state.location = self.geolocator.geocode("Nyeri")
        if "zoom" not in st.session_state:
            st.session_state["zoom"] = 18
        if "result" not in st.session_state:
            st.session_state["result"] = None
        if "message" not in st.session_state:
            st.session_state.message = []
        if "toast_iterations" not in st.session_state:
            st.session_state.toast_iterations = 0
        if "data_info_placeholder" not in st.session_state:
            st.session_state.data_info_placeholder = ""
        if "timer" not in st.session_state:
            st.session_state.timer = ""
        if "active_devices" not in st.session_state:
            st.session_state.active_devices = ""
        if "loaded" not in st.session_state:
            st.session_state.loaded = db_operations.load_data() 
        if "build_map" not in st.session_state:
            st.session_state.build_map = st.session_state.build_map = folium.Map(
                location=[st.session_state.location.latitude,  st.session_state.location.longitude],  
                zoom_start=st.session_state["zoom"],
                tiles=None,
                zoomControl=False
            )


    
    def reset_parameters(self):
        st.session_state.data_info_placeholder = ""
        st.session_state.build_mapodel_name = ""
        st.session_state["model_name_box"] = ""
        #del st.session_state["loaded"]
        #self.active_devices = db_operations.get_active_devices() 
 
    def start_data_collection(self):
        st.session_state.data_info_placeholder = f'''
        <span style="color:white">Collecting data…</span><br>
        <span style="color:lime">Available samples for training: {db_operations.get_no_of_rows(self.selected_devices)}</span>
        '''
        
    def add_learnmode_options(self):
        st.session_state.build_mapodel_name = st.sidebar.text_input(label="Add model name", on_change=self.start_data_collection, key="model_name_box")
        st.sidebar.markdown(st.session_state.data_info_placeholder, unsafe_allow_html=True)
        
    def plot_device(self, device:str):
        db_operations.get_device_data(device)
        db_operations.get_device_settings(device)
        locations = db_operations.get_device_positions(device)

        try:
            if st.session_state["show_polyline"]:
                folium.PolyLine(
                    locations = locations,
                    color = db_operations.get_device_settings(device)["Color_1"],
                    weight = 2.5,
                    opacity = 0.6
                ).add_to(st.session_state.build_map)

            for point in locations:
                folium.Circle(
                    location=[point[0], point[1]],  
                    radius=5,                    
                    color=db_operations.get_device_settings(device)["Color_1"],                  
                    fill=True,                   
                    fill_color=db_operations.get_device_settings(device)["Color_2"],             
                    fill_opacity=0.4,            
                    popup="Latitude: "+str(point[0])+" longitude: "+str(point[1]),
                
                ).add_to(st.session_state.build_map)

            popup_msg ="Device "+device+" Last seen Latitude: "+str(point[0])+", Longitude: "+str(point[1])
            mrker = folium.Marker([point[0], point[1]], icon=folium.Icon(color="red", icon="map-marker", icon_color="white")).add_to(st.session_state.build_map)
            mrker.add_child(folium.Popup(popup_msg))
        except:
            print("device data couldnt be loaded")

    def setup(self):
        st.session_state.build_map = st.session_state.build_map = folium.Map(
                location=[st.session_state.location.latitude,  st.session_state.location.longitude],  
                zoom_start=st.session_state["zoom"],
                tiles=None,
                zoomControl=False
            )
        self.place = st.sidebar.text_input(label="Find Location", placeholder="Search for a place", on_change=self.find, key="place_box")
        self.selected_devices = st.sidebar.multiselect("Devices", ["All"]+self.active_devices, placeholder="Select Devices", on_change=self.reset_parameters)
        st.sidebar.checkbox(label="Show path btwn points", key="show_polyline")
        mode = st.sidebar.selectbox("Mode", ["Monitor", "Train"], on_change=self.reset_parameters)

        if mode == "Train":
            self.add_learnmode_options()

        if "All" in self.selected_devices:
            self.selected_devices.extend(self.active_devices)
            self.selected_devices = list(dict.fromkeys(self.selected_devices))

        st.sidebar.button("Reload DB", on_click=db_operations.load_data)

        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI World Imagery",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(st.session_state.build_map)


        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI Reference Layer",
            name="Labels",
            overlay=True,
            control=True
        ).add_to(st.session_state.build_map)


        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI Roads",
            name="Roads",
            overlay=True,
            control=True
        ).add_to(st.session_state.build_map)

        folium.LayerControl().add_to(st.session_state.build_map)
        popup_msg = st.session_state.location.address + "\n\nLatitude: " + str(st.session_state.location.latitude) + ",\nLongitude: " + str(st.session_state.location.longitude)
        folium.Marker([st.session_state.location.latitude, st.session_state.location.longitude], popup=popup_msg ).add_to(st.session_state.build_map)
        

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
        for device in self.selected_devices:
            if device != "All":
                self.plot_device(device)            
    
        st.session_state["result"] = st_folium(st.session_state.build_map, height=800, width=None, returned_objects=[])
        if st.session_state["result"].get("last_clicked"):
            lat, lng = st.session_state["result"]["last_clicked"]["lat"], st.session_state["result"]["last_clicked"]["lng"]
            st.write("Last Click: ", lat, lng)

        '''
        if ((st.session_state["result"] and "zoom" in  st.session_state["result"])and
            st.session_state["result"] is not None):
            st.write(st.session_state["result"]["zoom"])
            st.session_state["zoom"] = st.session_state["result"]["zoom"]
            st.write(st.session_state["zoom"])
        '''
            
        

if __name__ == '__main__':
    app().run()