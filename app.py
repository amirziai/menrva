from flask import Flask, request, jsonify, render_template, Response
import redis
# from sklearn.externals import joblib as pickle
import _pickle as cPickle
import uuid
import os
import pandas as pd
import uuid as uuid_
import sqlite3
import json
from datetime import datetime
from functools import wraps


# TODO
# need to get data as JSON and not form data in upload

# sqlite database
# create table models (id text primary key, name text, description text, input text, output text, link text, timestamp datetime);
# redis flushall removes all keys

app = Flask(__name__)
db_path = 'db/models.sqlite'


def redis_server_connection():
	return redis.StrictRedis(host='localhost', port=6379, db=0)


@app.route('/', methods=['GET'])
def root():
	return render_template('index.html')


@app.route('/models', methods=['GET'])
def models():
	conn = sqlite3.connect(db_path)
	models = pd.read_sql_query('select * from models', conn).sort_values(by='timestamp', ascending=False)
	return jsonify({'status': 'ok', 'data': json.loads(models.to_json(orient='records'))})


def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'menrva' and password == 'menrva'


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
            'Could not verify your access level for that URL.\n'
            'You have to login with proper credentials', 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route('/upload', methods=['POST', 'GET'])
@requires_auth
def upload():
	if request.method == 'GET':
		return render_template('upload.html')

	file_uploaded = request.files['file']

	# find a unique uuid
	path = None
	while True:
		uuid = uuid_.uuid4()
		path = 'models/%s' % uuid

		if not os.path.exists(path):
			break


	file_uploaded.save(path)  # save to disk
	print ('%s: saved to disk') % uuid
	redis_server = redis_server_connection()

	# save to database
	json_ = dict(request.form)
	print (json_)
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	row = str(uuid), json_['name'][1], json_['description'][1], json_['input'][1], json_['output'][1], json_['link'][1], datetime.now()
	print (row)
	cursor.execute("insert into models values (?, ?, ?, ?, ?, ?, ?)", row)
	conn.commit()

	# print file_uploaded.read()
	# TODO: reading back from disk, couldn't get file_uploaded.read() working, produced ''
	redis_server.set(uuid, open(path).read())
	print ('%s: saved to memory (redis)') % uuid

	return jsonify({'status': 'ok', 'message': 'model was uploaded', 'uuid': uuid})


@app.route('/predict/<uuid>', methods=['POST'])
def predict(uuid):
	redis_server = redis_server_connection()

	# load the model into memory
	if redis_server.exists(uuid):
		# print redis_server.get(uuid)
		print ('Loading from redis')
		model = pickle.loads(redis_server.get(uuid))

	else:
		print ('%s: not found in memory') % uuid
		path = 'models/%s' % uuid

		if not os.path.exists(path):
			print ('%s: does not exist on disk (ERROR)') % uuid
			return jsonify({'status': 'error', 'message': 'model does not exist'})
		else:
			print ('%s: serialized to memory (redis)') % uuid
			model = pickle.loads(path)
			redis_server.set(uuid, open(path))  # serialize to memory


	try:
		json_ = request.json
		query = pd.DataFrame([json_])
		return jsonify({'prediction': list(model.predict(query))})

	except getopt.GetoptError as e:
		print ('error')
		print (e)
		return jsonify({'status': 'error', 'message': 'model prediction failed', 'error': e})


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, port=80)
