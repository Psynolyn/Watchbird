import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime
import db_operations

BROKERS = [
    {"host": "broker.hivemq.com", "port": 1883},
    {"host": "test.mosquitto.org", "port": 1883},
    {"host": "mqtt.eclipseprojects.io", "port": 1883}
]

TOPIC = "watchbird/device/location"
CLIENT_ID = f"watchbird_sub_{int(time.time())}"
KEEPALIVE = 120

current_broker_index = 0
message_count = 0
last_message_time = None

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        broker = BROKERS[current_broker_index]
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connected to {broker['host']}")
        print(f"Topic: {TOPIC}")
        print("=" * 70)
        
        client.subscribe(TOPIC, qos=1)
    else:
        print(f"[ERROR] Connection failed with code: {reason_code}")

def on_message(client, userdata, msg):
    global message_count, last_message_time
    
    try:
        message_count += 1
        last_message_time = datetime.now()
        
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        print(data)
        db_operations.write_to_data_table(data)
        
    except json.JSONDecodeError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Raw: {payload}")
    except Exception as e:
        print(f"[ERROR] {e}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    # Silent reconnection - no output
    pass

def try_connect():
    global current_broker_index
    
    for i, broker in enumerate(BROKERS):
        try:
            current_broker_index = i
            print(f"\nAttempting to connect to {broker['host']}:{broker['port']}...")
            
            client = mqtt.Client(
                client_id=CLIENT_ID,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                clean_session=False,
                protocol=mqtt.MQTTv311
            )
            
            client.on_connect = on_connect
            client.on_message = on_message
            client.on_disconnect = on_disconnect
            
            client.reconnect_delay_set(min_delay=1, max_delay=10)
            client.connect(broker['host'], broker['port'], keepalive=KEEPALIVE)
            
            return client
            
        except Exception as e:
            print(f"Failed to connect to {broker['host']}: {e}")
            if i < len(BROKERS) - 1:
                print("Trying next broker...")
            continue
    
    return None

# Main execution
print("=" * 70)
print("MQTT Watchbird Location Subscriber")
print("=" * 70)

client = try_connect()

if client:
    try:
        print("\nListening... (Press Ctrl+C to stop)")
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(f"[STATS] Total messages received: {message_count}")
        if last_message_time:
            print(f"[STATS] Last message: {last_message_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\nShutting down...")
        client.disconnect()
        client.loop_stop()
        print("Disconnected.")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        client.disconnect()
        client.loop_stop()
else:
    print("\n[ERROR] Could not connect to any MQTT broker")
    print("Please check your internet connection or try again later")