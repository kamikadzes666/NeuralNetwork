from MQTT import MQTT
from pprint import pprint

class TrackableObject:
	counter = 0
	def __init__(self, objectID, centroid, emo, age, gender):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		self.timestamp = 0
		self.emo = dict(emo)
		self.age = int(age)
		self.gender = gender

		# incrementing objects counter
		TrackableObject.counter +=1
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

	# getters
	def get_gender(self):
		return self.gender

	def get_age(self):
		return self.age

	def get_id(self):
		return self.objectID

	def get_emo(self):
		return self.emo

	# function changing emotion
	def change_emo(self, new_emo_zip):
		new_emo_dict = dict(new_emo_zip)
		for categoria, emothion in self.emo.items():
			self.emo[categoria] = round((self.emo[categoria] + new_emo_dict[categoria]) \
								  / 2,2)

	# time setter
	def set_time(self, time):
		self.timestamp = time

	# preparing data to sending to server
	def preparing_data(self):
		data_dict = {
			'loc_id': 0,
			'box_mac': 0,
			'user_fid': self.objectID,
			'sex': self.gender,
			'age': self.age,
			'emotion': self.emo,
			'time': self.timestamp
		}
		# print(data_dict)
		return data_dict

	def send_data(self,):
		# client = MQTT()
		data_to_send = self.preparing_data()
		# client.connection(data_to_send)
		return data_to_send
