import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import db_operations
import train
from dataset import db_csv
import os
import pandas as pd
from datetime import datetime

class app:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="streamlit_map")
        st.set_page_config(layout="wide")
        self.m = ""
        self.selected_devices = []
        self.messages = ["Logger.get_logs()"]
        self.dev_mode = 1
        
        
        if "loaded" not in st.session_state:
            st.session_state.loaded = db_operations.load_data() 
        if "location" not in st.session_state:
            st.session_state.location = self.geolocator.geocode("Nairobi")
        if "zoom" not in st.session_state:
            st.session_state["zoom"] = 18
        if "model_name" not in st.session_state:
            st.session_state["model_name"] = None
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
        if "last_processed_ids" not in st.session_state:
            # Track last processed record ID per device to detect new data
            st.session_state.last_processed_ids = {}
        if "monitoring_stats" not in st.session_state:
            # Store monitoring statistics
            st.session_state.monitoring_stats = {
                'total_processed': 0,
                'anomalies_detected': 0,
                'last_check': None
            }
        
       
    
    def reset_parameters(self):
        st.session_state.data_info_placeholder = ""
        self.model_name = ""
        st.session_state["model_name_box"] = ""
        db_operations.reload_db()
        if len(st.session_state["multiselect"]) > 1 and "All" in self.selected_devices:
            self.selected_devices.remove("All")
            prev = st.session_state["multiselect"]
            st.session_state["multiselect"] = []
            new = []
            for i in prev:
                if i != "All":
                    new.append(i)
            st.session_state["multiselect"] = new

    def start_data_collection(self):
        st.session_state.data_info_placeholder = f'''
        <span style="color:white">Collecting data…</span><br>
        <span style="color:lime">Available samples for training: {db_operations.get_no_of_rows(self.selected_devices)}</span>
        '''

    def start_training(self):
        if len(self.selected_devices) != 0:
            db_operations.reload_db()
            db_csv.prepare_dataset()
            train.run_pipeline(
                device_ids=[db_operations.device_name_to_id.get(device) for device in self.selected_devices],
                perform_hp_search=False,
                create_visualizations=True,
                model_name=st.session_state["model_name"]
            )
            trained_devices = [{"Device_name":device, "IF_model":st.session_state["model_name"]} for device in self.selected_devices if device != "All"]
            for model in trained_devices:
                db_operations.save_device_if_model(model)
            
            db_operations.reload_db()
            allowed_files = []
            for model in db_operations.available_models:
                allowed_files.append(model+".pkl")
                allowed_files.append(model+"_Scaler.pkl")

            for filename in os.listdir("models"):
                if filename not in allowed_files:    
                    full_path = os.path.join("models", filename)
                    os.remove(full_path)

    def reload_and_monitor(self):
        """
        Reload database and run anomaly detection on new data points.
        This is called when user clicks "Reload DB" in Monitor mode.
        """
        # Reload the database
        db_operations.reload_db()
        
        # Check if we're in monitor mode and have devices selected
        if len(self.selected_devices) == 0:
            st.session_state.message = ["info", "No devices selected for monitoring"]
            return
        
        # Get devices to monitor (exclude "All" label)
        devices_to_monitor = [d for d in self.selected_devices if d != "All"]
        
        total_new_points = 0
        total_anomalies = 0
        devices_with_anomalies = []
        
        # Process each device
        for device in devices_to_monitor:
            # Get the assigned model for this device
            model_name = db_operations.get_device_model(device)
            
            if not model_name:
                st.warning(f"Device '{device}' has no model assigned. Assign a model in 'Device Model Assignment' mode.")
                continue
            
            # Get new data points for this device
            new_data = self.get_new_data_for_device(device)
            
            if new_data is None or len(new_data) == 0:
                continue
            
            total_new_points += len(new_data)
            
            # Run inference on new data
            try:
                anomalies_found = self.run_inference_on_device(device, model_name, new_data)
                
                if anomalies_found > 0:
                    total_anomalies += anomalies_found
                    devices_with_anomalies.append(f"{device} ({anomalies_found})")
                    
            except Exception as e:
                st.error(f"Error processing device '{device}': {str(e)}")
        
        # Update monitoring stats
        st.session_state.monitoring_stats['total_processed'] += total_new_points
        st.session_state.monitoring_stats['anomalies_detected'] += total_anomalies
        st.session_state.monitoring_stats['last_check'] = datetime.now()
        
    def get_new_data_for_device(self, device: str) -> pd.DataFrame:
        """
        Get only NEW data points for a device that haven't been processed yet.
        
        Returns:
            DataFrame with new unprocessed data points
        """
        device_id = db_operations.device_name_to_id.get(device)
        
        if device_id is None:
            return None
        
        # OPTION A: Real-time from Supabase (queries database directly)
        # Uncomment this to always get latest data:
        # all_data = db_operations.get_device_data_for_inference(device)
        
        # OPTION B: From cached data (faster, but needs reload_db() first)
        # This uses data already loaded in memory
        all_data = db_operations.get_device_data_for_inference_cached(device)
        
        if all_data is None or len(all_data) == 0:
            return None
        
        # Get last processed ID for this device
        last_id = st.session_state.last_processed_ids.get(device, 0)
        
        # Filter to only new records
        new_data = all_data[all_data['Id'] > last_id].copy()
        
        # Update last processed ID
        if len(new_data) > 0:
            st.session_state.last_processed_ids[device] = new_data['Id'].max()
        
        return new_data

    def run_inference_on_device(self, device: str, model_name: str, new_data: pd.DataFrame) -> int:
        """
        Run anomaly detection inference on new data points.
        
        Args:
            device: Device name
            model_name: Name of the model to use
            new_data: DataFrame with new data points
        
        Returns:
            Number of anomalies detected
        """
        if len(new_data) == 0:
            return 0
        
        # Load the model
        model_path = f'models/{model_name}.pkl'
        scaler_path = f'models/{model_name}_Scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error(f"Model '{model_name}' files not found!")
            return 0
        
        try:
            model, scaler, feature_columns = train.load_model(model_path, scaler_path)
        except Exception as e:
            st.error(f"Error loading model '{model_name}': {str(e)}")
            return 0
        
        # Prepare data for inference
        # Make sure new_data has all required columns
        df_prepared = self.prepare_data_for_inference(new_data)
        
        # Engineer features
        df_features, _ = train.build_features(df_prepared)
        
        # Extract feature matrix
        X = df_features[feature_columns].values
        
        # Score and label
        scores, labels = train.score_and_label(model, scaler, X)
        
        # Add predictions to dataframe
        df_features['anomaly_score'] = scores
        df_features['Anomaly'] = labels
        
        # Filter isolated anomalies (optional but recommended)
        df_filtered = train.filter_isolated_anomalies(
            df_features,
            gap_threshold=train.Config.EVENT_GAP_THRESHOLD,
            min_event_size=train.Config.MIN_EVENT_SIZE
        )
        
        # Count anomalies after filtering
        n_anomalies = (df_filtered['Anomaly'] == -1).sum()
        
        # Save results back to database
        if n_anomalies > 0:
            # Prepare data for database update
            results_to_save = df_filtered[['Id', 'Anomaly']].copy()
            results_to_save['Id'] = results_to_save['Id'].astype(int)
            results_to_save['Anomaly'] = results_to_save['Anomaly'].astype(int)
            
            # Save to database
            db_operations.save_outlier_result(results_to_save.to_dict(orient="records"))
        
        return n_anomalies

    def prepare_data_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for inference by ensuring all required columns exist.
        
        Args:
            df: Raw data from database
        
        Returns:
            Prepared DataFrame ready for feature engineering
        """
        df_prep = df.copy()
        
        # Ensure Timestamp is datetime
        if 'Timestamp' in df_prep.columns:
            df_prep['Timestamp'] = pd.to_datetime(df_prep['Timestamp'], utc=True)
        
        # Add any missing columns with default values
        required_cols = ['Device_id', 'Latitude', 'Longitude', 'Altitude', 'Timestamp']
        for col in required_cols:
            if col not in df_prep.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Sort by timestamp (important for time-series features)
        df_prep = df_prep.sort_values('Timestamp').reset_index(drop=True)
        
        return df_prep

    def add_monitor_mode_sidebar(self):
        """
        Add monitoring statistics and controls to sidebar in Monitor mode.
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader("Monitoring Stats")
        
        stats = st.session_state.monitoring_stats
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Processed", stats['total_processed'])

        if stats['last_check']:
            last_check_str = stats['last_check'].strftime('%H:%M:%S')
            st.sidebar.text(f"Last check: {last_check_str}")
        
        # Show device statuses
        if self.selected_devices and len([d for d in self.selected_devices if d != "All"]) > 0:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔍 Device Status")
            
            for device in self.selected_devices:
                if device == "All":
                    continue
                
                model_name = db_operations.get_device_model(device)
                
                if model_name:
                    st.sidebar.text(f"✅ {device}")
                    st.sidebar.text(f"   Model: {model_name}")
                else:
                    st.sidebar.text(f"⚠️ {device}")
                    st.sidebar.text(f"   No model assigned")
        
        # Auto-refresh option (optional)
        st.sidebar.markdown("---")
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()

    def add_learnmode_options(self):
        st.session_state["model_name"] = st.sidebar.text_input(
            label="Add new model name", 
            on_change=self.start_data_collection, 
            key="model_name_box"
        ).strip()
        st.sidebar.markdown(st.session_state.data_info_placeholder, unsafe_allow_html=True)
        if st.session_state["model_name"] or st.session_state["model_name"] != "":
            st.sidebar.button("Train", on_click=self.start_training)
    
    def add_model_assiggnment_options(self):
        st.sidebar.text("Assign models to your devices")
        
        if "All" not in self.selected_devices:
            for device in self.selected_devices:
                key = f"model_{device}"
                if key not in st.session_state:
                    st.session_state[key] = db_operations.get_device_model(device)
        
                try:
                    index = db_operations.available_models.index(db_operations.get_device_model(device))
                except:
                    index = None

                selected = st.sidebar.selectbox(
                    device, 
                    db_operations.available_models, 
                    placeholder="Choose a model",
                    index=index,
                    key=f"model_{device}"
                )
                current_model = db_operations.get_device_model(device)
                if selected and selected != current_model:
                    db_operations.save_device_if_model({
                        "Device_name": device,
                        "IF_model": selected
                    })
                                                
        else:
            changed = False
            selected = st.sidebar.selectbox(
                "All", 
                db_operations.available_models, 
                index=None, 
                placeholder="Choose a model",
            )
            if selected:
                for device in db_operations.get_active_devices():
                    db_operations.save_device_if_model({
                        "Device_name": device,
                        "IF_model": selected
                    })

    def plot_device(self, device: str):
        db_operations.get_device_data(device)
        db_operations.get_device_settings(device)
        locations = db_operations.get_device_positions(device)

        try:
            if st.session_state["show_polyline"]:
                folium.PolyLine(
                    locations=db_operations.get_device_polyline_points(device),
                    color=db_operations.get_device_settings(device)["Color_1"],
                    weight=2.5,
                    opacity=0.6
                ).add_to(self.m)

            for point in locations:
                if point[2] == -1:
                    color_1 = "red"
                    color_2 = "red"
                    radius = 160
                else:
                    color_1 = db_operations.get_device_settings(device)["Color_1"]
                    color_2 = db_operations.get_device_settings(device)["Color_2"]
                    radius = 5

                folium.Circle(
                    location=[point[0], point[1]],  
                    radius=radius,                    
                    color=color_1,                  
                    fill=True,                   
                    fill_color=color_2,             
                    fill_opacity=0.4,            
                    popup="Latitude: " + str(point[0]) + " longitude: " + str(point[1]),
                ).add_to(self.m)

            popup_msg = "Device " + device + " Last seen Latitude: " + str(point[0]) + ", Longitude: " + str(point[1])
            mrker = folium.Marker(
                [point[0], point[1]], 
                icon=folium.Icon(color="red", icon="map-marker", icon_color="white")
            ).add_to(self.m)
            mrker.add_child(folium.Popup(popup_msg))
        except:
            print("device data couldn't be loaded")

    def setup(self):
        self.m = folium.Map(
            location=[st.session_state.location.latitude, st.session_state.location.longitude],  
            zoom_start=st.session_state["zoom"],
            tiles=None,
        )
        self.place = st.sidebar.text_input(
            label="Find Location", 
            placeholder="Search for a place", 
            on_change=self.find, 
            key="place_box"
        )
        self.selected_devices = st.sidebar.multiselect(
            "Devices", 
            ["All"] + db_operations.get_active_devices(), 
            placeholder="Select Devices", 
            on_change=self.reset_parameters, 
            key="multiselect"
        )
        st.sidebar.checkbox(label="Show path btwn points", key="show_polyline")
        
        if self.dev_mode:
            modes = ["Monitor", "Train", "Device Model Assignment"]
        else:
            modes = ["Monitor", "Device Model Assignment"]

        mode = st.sidebar.selectbox("Mode", modes, on_change=self.reset_parameters)

        # Add mode-specific sidebar options
        if mode == "Monitor":
            self.add_monitor_mode_sidebar()
        elif mode == "Train":
            self.add_learnmode_options()
        elif mode == "Device Model Assignment":
            self.add_model_assiggnment_options()

        if "All" in self.selected_devices:
            self.selected_devices.extend(db_operations.get_active_devices())
            self.selected_devices = list(dict.fromkeys(self.selected_devices))

        # Modify Reload DB button based on mode
        if mode == "Monitor":
            st.sidebar.button("Reload & Monitor", on_click=self.reload_and_monitor)
        else:
            st.sidebar.button("Reload DB", on_click=db_operations.reload_db)

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
        popup_msg = (st.session_state.location.address + "\n\nLatitude: " + 
                    str(st.session_state.location.latitude) + 
                    ",\nLongitude: " + str(st.session_state.location.longitude))
        folium.Marker(
            [st.session_state.location.latitude, st.session_state.location.longitude], 
            popup=popup_msg
        ).add_to(self.m)

        if st.session_state.message != []:
            st.session_state.toast_iterations += 1
            st.toast(st.session_state.message[1])

            if st.session_state.toast_iterations >= 3:
                st.session_state.toast_iterations = 0
                st.session_state.message = []
        
    def find(self):
        query = st.session_state["place_box"].strip()
        if query:
            temp_location = self.geolocator.geocode(query)
            if temp_location:
                st.session_state.location = temp_location
                st.session_state.message = ["success", st.session_state.location.address]
            else:
                st.session_state.message = ["error", "Place not found"]
        else:
            pass

    def run(self):
        self.setup()
        for device in self.selected_devices:
            if device != "All":
                self.plot_device(device)            
    
        st_folium(self.m, height=800, width=None, returned_objects=[])

if __name__ == '__main__':
    app().run()