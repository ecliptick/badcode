import os
import requests

from celery import Celery
from celery.utils.log import get_task_logger

from anpr import *

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

CORE_HOSTNAME = os.getenv("CORE_HOSTNAME")

logger = get_task_logger(__name__)

app = Celery("tasks", broker="amqp://admin:mypass@rabbit:5672", backend="rpc://")


@app.task(serializer="json")
def process(
    cam_id,
    lane_number,
    input_path,
    output_path,
    rejected_path,
    filename,
    incident_id,
    cam_location_code,
    cam_location_description,
):

    logger.info("Got Request - Starting work ")

    anpr = ANPR()

    result = anpr.perform_image_proc(
        input_path, output_path, rejected_path, filename, cam_location_code, lane_number
    )

    result["incidentId"] = incident_id
    result["camId"] = cam_id
    result["camLocationCode"] = cam_location_code
    result["camLocationDescription"] = cam_location_description

    logger.info("Work Finished ")

    try:
        response = requests.post(CORE_HOSTNAME + "/api/updateIncident", json=result)

        print(response.text)
        print(response.status_code)

    except Exception as err:
        print(err)
        pass

    return result
