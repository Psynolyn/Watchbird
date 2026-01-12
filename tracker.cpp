#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <queue>

std::queue<String> mqttpayloads;

const char* ssid     = "hott";
const char* password = "hotthott";

const char* broker = "broker.hivemq.com";
const int port = 1883;
const char* topic = "watchbird/device/location";

WiFiClient espClient;
PubSubClient mqtt(espClient);

bool show_debug_msgs = 1;

uint32_t last_blink = 0, last_publish = -15000;
bool led_status = 1;
int Device_id = 4, Latitude=0, Longitude=0, Altitude=0;

void setup() {
  
  Serial.begin(115200);
  delay(100);

  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  if(show_debug_msgs){Serial.println(" connected!");}

  mqtt.setServer(broker, port);

  configTime(3 * 3600, 0, "pool.ntp.org", "time.nist.gov");
  Serial.println("Waiting for NTP time...");
  while (time(nullptr) < 1600000000) {
    delay(500);
  }

  pinMode(2, OUTPUT);
}

void loop() {
  if (!mqtt.connected()) {
    if(show_debug_msgs){Serial.print("Connecting to MQTT...");}
    while (!mqtt.connected()) {
      if (mqtt.connect("ESP32WiFiClient")) {
        if(show_debug_msgs){Serial.println(" connected!");}
      } else {
        if(show_debug_msgs){
          Serial.print(" failed, rc=");
          Serial.print(mqtt.state());
          Serial.println(" trying again in 5s");
        }
        delay(5000);
      }
    }
  }
  
  
  if(mqtt.connected() && (millis() - last_blink) > 1000){
    last_blink = millis();
    led_status = !led_status;
    digitalWrite(2, led_status);
  }

  if ((millis() - last_publish) > 15000){
    last_publish = millis();
    time_t now = time(nullptr);
    char time_buf[30];
    struct tm* tm_info = localtime(&now);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%S+03:00", tm_info);

    StaticJsonDocument<200U> doc;
    doc["Timestamp"] = time_buf;
    doc["Device_id"] = Device_id;
    doc["Latitude"] = Latitude;
    doc["Longitude"] = Longitude;
    doc["Altitude"] = Altitude;

    char buffer[256];
    size_t n = serializeJson(doc, buffer);
    mqttpayloads.push(String(buffer));
    }

    while(mqtt.connected() && !mqttpayloads.empty()){
      String payload = mqttpayloads.front();
      mqtt.publish(topic, payload.c_str());
      if(show_debug_msgs){
      Serial.print("Payload sent");
      Serial.print(mqttpayloads.front());
      }
      mqttpayloads.pop();
  }

}