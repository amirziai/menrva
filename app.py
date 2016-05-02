from flask import Flask, request, jsonify
import redis
# from sklearn.externals import joblib as pickle
import cPickle as pickle
import uuid
import os
import pandas as pd
import uuid as uuid_


app = Flask(__name__)

# TODOs
# 1- add a dropzone UI to drag and drop pickled models
# 2- get a list of all models (in redis and on disk)
# 3- Log usage of models

def redis_server_connection():
	return redis.StrictRedis(host='localhost', port=6379, db=0)


@app.route('/', methods=['GET'])
def root():
	return jsonify({'status': 'ok'})


@app.route('/upload', methods=['POST'])
def upload():
	file_uploaded = request.files['file']

	# find a unique uuid
	path = None
	while True:
		uuid = uuid_.uuid4()
		path = 'models/%s' % uuid

		if not os.path.exists(path):
			break


	file_uploaded.save(path)  # save to disk
	print '%s: saved to disk' % uuid
	redis_server = redis_server_connection()
	print file_uploaded.read()
	# TODO: reading back from disk, couldn't get file_uploaded.read() working, produced ''
	redis_server.set(uuid, open(path).read())
	print '%s: saved to memory (redis)' % uuid

	return jsonify({'status': 'ok', 'message': 'model was uploaded', 'uuid': uuid})


@app.route('/predict/<uuid>', methods=['POST'])
def predict(uuid):
	redis_server = redis_server_connection()
	
	# load the model into memory
	if redis_server.exists(uuid):
		print redis_server.get(uuid)
		model = pickle.loads(redis_server.get(uuid))

	else:
		print '%s: not found in memory' % uuid
		path = 'models/%s' % uuid

		if not os.path.exists(path):
			print '%s: does not exist on disk (ERROR)' % uuid
			return jsonify({'status': 'error', 'message': 'model does not exist'})
		else:
			print '%s: serialized to memory (redis)' % uuid 
			model = pickle.loads(path)
			redis_server.set(uuid, open(path))  # serialize to memory


	try:
		json_ = request.json
		query = pd.DataFrame([json_])
		return jsonify({'prediction': list(model.predict(query))})

	except Exception, e:
		print 'error'
		print e
		return jsonify({'status': 'error', 'message': 'model prediction failed'})


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)