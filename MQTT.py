import paho.mqtt.client as paho
import json

# MQTTTESTSERVER = 'test.mosquitto.org'
MQTTSERVER = 'mqtt.wifly.net'
MQTTPORT = 1883
TOPICNAME = 'neurobox/cc50e3dac77d'
MQTTLOGIN = 'flyEyeNeuroBox'
MQTTPASS = 'be0y#2hs&@i875'


class MQTT:
    def __init__(self):
        self.client = paho.Client("test")
        self.client.on_message = self.on_message
        self.client.on_log = self.on_log
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.username_pw_set(MQTTLOGIN, password=MQTTPASS)

    # define callbacks
    def on_message(self, client, userdata, message):
        print("received message =", str(message.payload.decode("utf-8")))

    def on_log(self, client, userdata, level, buf):
        print("log: ", buf)

    def on_connect(self, client, userdata, flags, rc):
        print("publishing ")

    def on_publish(self, client, userdata, result):
        print("data published \n")

    def connection(self, msg):
        self.client.connect(MQTTSERVER, MQTTPORT, 60)
        self.client.loop_start()
        # publish and logging and exit
        msg_json = json.dumps(msg)
        # print publishing data if you need it
        # pprint(msg_json)
        self.client.publish(TOPICNAME, msg_json)
        self.client.disconnect()
        self.client.loop_stop()
