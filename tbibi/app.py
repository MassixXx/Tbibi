from flask import Flask, request, abort, jsonify
import json
import numpy as np
from detection_models import predict_burn, predict_skin_illness

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict_skin():
	print('enter endpoint')
	files = request.files
	if len(files)==0:
		print('Erreur lors de la réception du fichier')
		return {"success : False"}
	for x in files:
		files[x].save('uploads/' + files[x].filename)
		r = predict_burn('uploads/' + files[x].filename)
		if r['detected']:
				return jsonify(
					{
					"detected": True,
					"type" : 'burn',
					"degree" : r['degree'],
					"emergency" : True
					})
		else :
				r = predict_skin_illness('uploads/' + files[x].filename)
				return jsonify(r)

