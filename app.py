import flask
import json
import warnings
import traceback


from flask import Response, request
from flask_cors import CORS
from Algorithm.Sentiment_analysis import train_model, make_prediction

warnings.simplefilter(action='ignore', category=FutureWarning)

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

            Y_predicted, Y_real, dates = train_model(currency_o, currency_d, fecha_ini, fecha_fin)

            #print(dates)

            global trained
            trained = True

            service_response = {'Predicted_values': Y_predicted.ravel().tolist(),
                                'Real_values': Y_real.ravel().tolist(), 'Dates': dates}

            response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)

        except Exception as ex:
            print(traceback.format_exc())
            response = Response("Error processing", 500)

    return response


@application.route("/predict", methods=["POST"])
def predict():
    global trained
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            if trained:
                request_content = json.loads(request.data)
                message = request_content
                days_to_predict = message["days_to_predict"]
                y_pred, _dates = make_prediction(prediction_days=days_to_predict)
                service_response = {'Predicted_values': y_pred.tolist(), 'Dates': _dates.tolist()}
                response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)
            else:
                response = Response("Call the training model method first", 405)
        except Exception as ex:
            print(traceback.format_exc())
            response = Response("Error processing", 500)
    return response


if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=True)
