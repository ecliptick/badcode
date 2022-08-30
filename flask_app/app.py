try:
    import os
    import cv2
    import numpy as np
    import paramiko
    from flask import Flask, Response, request
    from celery import Celery
    
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())

    FTP_HOSTNAME = os.getenv('FTP_HOSTNAME')
    FTP_USERNAME = os.getenv('FTP_USERNAME')
    FTP_PASSWORD = os.getenv('FTP_PASSWORD')
    
    INPUT_PATH = os.getenv('INPUT_PATH')
    OUTPUT_PATH = os.getenv('OUTPUT_PATH')
    FILENAME = os.getenv('FILENAME')    
    
except Exception as e:
    print("Error  :{} ".format(e))

app = Flask(__name__)

simple_app = Celery(
    "simple_worker", broker="amqp://admin:mypass@rabbit:5672", backend="rpc://"
)

@app.route("/api/speedcamera", methods=["POST"])
def call_method():
    data = request.get_json()

    app.logger.info("Invoking Method ")

    if data is None:
        error = "Content is empty"
        Response(error, status=406)

    try:
        # data["camId"]
        data["laneNumber"]
        data["path"]
        data["outputPath"]
        data["rejectedPath"]
        data["filename"]
        data["incidentId"]
        data["camLocationCode"]
        data["camLocationDescription"]

    except Exception as e:

        error = str(e) + "  not found"

        return Response(error, status=406)

    r = simple_app.send_task(
        "tasks.process",
        kwargs={
            "cam_id": data["camId"],
            "lane_number": data["laneNumber"],
            "input_path": data["path"],
            "output_path": data["outputPath"],
            "rejected_path": data["rejectedPath"],
            "filename": data["filename"],
            "incident_id": data["incidentId"],
            "cam_location_code": data["camLocationCode"],
            "cam_location_description": data["camLocationDescription"]
        },
    )

    app.logger.info(r.backend)

    message = "Data received with id = " + str(r.id)

    return Response(message)