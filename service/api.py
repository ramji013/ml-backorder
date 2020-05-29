from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

from flasgger import Swagger


app = Flask(__name__)
Swagger(app)
pickle_in = open('../model/classifier.pkl', 'rb')

classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return 'welcome all'


@app.route('/predict',methods=["Get"])
def predict_for_single():

    """
        This application will predict whether the particular product go backorder or not
    ---
    parameters:

        # - name: product_random_id
        #   in: query
        #   type: number
        #   required: true

        - name: current_inventory_level
          in: query
          type: number
          required: true

        - name: transit_time
          in: query
          type: number
          required: true

        - name: in_transit_qty
          in: query
          type: number
          required: true

        - name: forecast_3_month
          in: query
          type: number
          required: true

        - name: forecast_6_month
          in: query
          type: number
          required: true

        - name: forecast_9_month
          in: query
          type: number
          required: true

        - name: sales_1_month
          in: query
          type: number
          required: true

        - name: sales_3_month
          in: query
          type: number
          required: true

        - name: sales_6_month
          in: query
          type: number
          required: true

        - name: sales_9_month
          in: query
          type: number
          required: true

        - name: min_bank
          in: query
          type: number
          required: true

        - name: pieces_past_due
          in: query
          type: number
          required: true

        - name: perf_6_month_avg
          in: query
          type: number
          required: true

        - name: perf_12_month_avg
          in: query
          type: number
          required: true

    responses:
        200:
            description: The output values

    """

    #product_random_id = float(request.args.get('product_random_id'))
    current_inventory_level = float(request.args.get('current_inventory_level'))
    transit_time = float(request.args.get('transit_time'))
    in_transit_qty = float(request.args.get('in_transit_qty'))
    forecast_3_month = float(request.args.get('forecast_3_month'))
    forecast_6_month = float(request.args.get('forecast_6_month'))
    forecast_9_month = float(request.args.get('forecast_9_month'))
    sales_1_month = float(request.args.get('sales_1_month'))
    sales_3_month = float(request.args.get('sales_3_month'))
    sales_6_month = float(request.args.get('sales_6_month'))
    sales_9_month = float(request.args.get('sales_9_month'))
    min_stock = float(request.args.get('min_bank'))
    pieces_past_due = float(request.args.get('pieces_past_due'))
    perf_6_month_avg = float(request.args.get('perf_6_month_avg'))
    perf_12_month_avg = float(request.args.get('perf_12_month_avg'))


    # product_random_id = request.args.get('product_random_id')
    # current_inventory_level = request.args.get('current_inventory_level')
    # transit_time = request.args.get('transit_time')
    # in_transit_qty = request.args.get('in_transit_qty')
    # forecast_3_month = request.args.get('forecast_3_month')
    # forecast_6_month = request.args.get('forecast_6_month')
    # forecast_9_month = request.args.get('forecast_9_month')
    # sales_1_month = request.args.get('sales_1_month')
    # sales_3_month = request.args.get('sales_3_month')
    # sales_6_month = request.args.get('sales_6_month')
    # sales_9_month = request.args.get('sales_9_month')
    # min_stock = request.args.get('min_bank')
    # pieces_past_due = request.args.get('pieces_past_due')
    # perf_6_month_avg = request.args.get('perf_6_month_avg')
    # perf_12_month_avg = request.args.get('perf_12_month_avg')

    prediction = classifier.predict([[current_inventory_level, transit_time, in_transit_qty,
                                      forecast_3_month, forecast_6_month, forecast_9_month, sales_1_month, sales_3_month,
                                      sales_6_month, sales_9_month, min_stock, pieces_past_due, perf_6_month_avg, perf_12_month_avg]])

    print("predicted value : " + str(prediction))

    return "The predicted value for the given input is " + str(prediction)

if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
#     app.run(host= '0.0.0.0', port=8080)