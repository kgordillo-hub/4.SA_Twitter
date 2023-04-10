import flask
import json
import warnings
import traceback

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import base64

from flask import Response, request
from flask_cors import CORS
from io import StringIO
from Algorithm.Sentiment_analysis import train_model

application = flask.Flask(__name__)
CORS(application)

trained = False


@application.route("/trainModel", methods=["POST"])
def train():
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            request_content = json.loads(request.data)
            message = request_content

            print("Training - JSON content ", message)

            currency_o = message['currency_o']
            currency_d = message['currency_d']

            fecha_ini = message['date_ini']
            fecha_fin = message['date_fin']

            # data = message['data']
            # csv_decoded = base64.b64decode(data).decode('utf-8')
            # df = pd.read_csv(StringIO(csv_decoded, newline=''), delimiter=',')

            Y_predicted, Y_real, dates = train_model(currency_o, currency_d, fecha_ini, fecha_fin)

            global trained
            trained = True

            service_response = {'Predicted_values': Y_predicted.ravel().tolist(),
                                'Real_values': Y_real.ravel().tolist(), 'Dates': dates.tolist()}

            response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)

        except Exception as ex:
            print(traceback.format_exc())
            response = Response("Error processing", 500)

    return response


def addDAys(dates):
    for i in range(len(dates)):
        new_date = pd.to_datetime(dates[i]) + pd.DateOffset(days=i + 1)
        dates[i] = new_date

    return dates


# @application.route("/predict", methods=["POST"])
# def predict():
#    global trained
#    if request.json is None:
#        # Expect application/json request
#        response = Response("Empty request", status=415)
#    else:
#        try:
#            if trained:
#                request_content = json.loads(request.data)
#                message = request_content
#                days_to_predict = message["days_to_predict"]
#                #print("Predict - JSON content ", message)
#
#                #make_prediction(prediction_days=days_to_predict)
#
#                y_pred, _dates = make_prediction(prediction_days=days_to_predict)
#
#                #print("Prediction: ", y_pred)
#                #print("Dates: ", _dates)
#                service_response = {'Predicted_values': y_pred.tolist(), 'Dates': _dates.tolist()}
#                response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)
#            else:
#                response = Response("Call the training model method first", 405)
#        except Exception as ex:
#            print(traceback.format_exc())
#            response = Response("Error processing", 500)
#
#    return response


if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=True)
