import os
import cv2

import traceback
import matplotlib.pyplot as plt
from car import Vehicle
from enhance import perform_image_enhancement, perform_image_enhancement_f, abright, perform_abright
from ocr import OCR
from plate import Plate
from ocr_rule import ocr_validation
from dotenv import load_dotenv, find_dotenv
import numpy as np

load_dotenv(find_dotenv())

FTP_HOSTNAME = os.getenv("FTP_HOSTNAME")
FTP_USERNAME = os.getenv("FTP_USERNAME")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")


class ANPR:
    xm = None
    ym = None

    def __init__(self):
        self.ocr = OCR()
        self.plate = Plate()
        self.vehicle = Vehicle()

    def read_image(self, file_path, filename):

        image = cv2.imread(file_path + filename)

        if image.shape[0] == 0:
            raise Exception("Image not found or input are wrong")

        return image

    def write_image(self, image, filename, upload_path):

        cv2.imwrite(upload_path + filename, image)

    def assigner(self, x_m, y_m, x_adj, y_adj):

        x1_new = x_m - x_adj
        x2_new = x_m + x_adj

        y1_new = y_m - y_adj
        y2_new = y_m + y_adj

        return x1_new, x2_new, y1_new, y2_new

    def generate_tail_light_image(self, image):

        x1_new = self.vehicle.x + self.plate.xmin
        y1_new = self.vehicle.y + self.plate.ymin
        x2_new = self.vehicle.x + self.plate.xmax
        y2_new = self.vehicle.y + self.plate.ymax

        # print("mids from raws xyxy")
        # print(x1_new,y1_new,x2_new,y2_new)

        # get crop of original v4 for ratio
        w_new = self.vehicle.w
        h_new = self.vehicle.h

        x_m = int((x1_new + x2_new) * 0.5)
        y_m = int((y1_new + y2_new) * 0.5)

        is_lower = None
        #
        if int(self.vehicle.y + 0.6 * (h_new)) < y_m:
            is_lower = True

        # print(f"mid of lp is {y_m}, mid of whole is {int(self.vehicle.y + 0.5 * (h_new))}")

        self.xm = x_m
        self.ym = y_m

        # print("done mids")
        # print(x_m,y_m)
        # print(f"from plate xxyy {self.plate.xmin,self.plate.xmax,self.plate.ymin,self.plate.ymax}")
        # moto, human and bike
        if self.vehicle.vehicle_type == 2:
            # pure plate from img, skipping v4 crop step
            # print("norm car plate")

            x_adj = int(w_new * 0.27)
            # no need y_adj since ratio side will handle h
            y_adj = 1
            # y_adj = int(h_new * 0.8)

            # we can add slight offset here, but both x_adj sum must = 2
            x11 = x_m - int(1.025 * x_adj)
            # print(f"x11 is {x11}")
            x22 = x_m + int(0.975 * x_adj)

            y11 = y_m - y_adj
            y22 = y_m + y_adj

            # y1_new = y1_new - y_adj
            # y2_new = y2_new + y_adj

        elif self.vehicle.vehicle_type == 6 or self.vehicle.vehicle_type == 7:
            # print("trigger truck n trains")
            x_adj = int(w_new * 0.4)
            # no need y_adj since ratio side will handle h
            y_adj = 1

            # x1_new = x1_new - x_adj
            # x2_new = x2_new + x_adj

            x11 = x_m - int(1.025 * x_adj)
            x22 = x_m + int(0.975 * x_adj)

            # y1_new = y_m - y_adj
            # y2_new = y_m + y_adj

            y11 = y_m - y_adj
            y22 = y_m + y_adj

            # x1_new = self.vehicle.x + self.plate.xmin * 0.4
            # y1_new = self.vehicle.y + self.plate.ymin * 0.4
            # x2_new = self.vehicle.x + self.plate.xmax * 1.1
            # y2_new = self.vehicle.y + self.plate.ymax * 1.1
        # bus
        elif self.vehicle.vehicle_type == 5:
            # print("trigger bus")
            x_adj = int(w_new * 0.28)
            # no need y_adj since ratio side will handle h
            y_adj = 1

            # x1_new = x1_new - x_adj
            # x2_new = x2_new + x_adj

            x11 = x_m - int(1.03 * x_adj)
            x22 = x_m + int(0.97 * x_adj)

            # y1_new = y_m - y_adj
            # y2_new = y_m + y_adj

            # y1_new = y1_new - y_adj
            # y2_new = y2_new + y_adj

            y11 = y_m - y_adj
            y22 = y_m + y_adj

        elif self.vehicle.vehicle_type == 0 or 1 or 3:
            # print("lp moto")
            # x_adj = int((x2_new - x1_new) * 0.1)
            # y_adj = int((y2_new - y1_new) * 0.1)
            #
            x_adj = int(w_new * 0.2)
            # no need y_adj since ratio side will handle h

            y_adj = 1
            # y_adj = int(x_adj/1.4)
            # y_adj = int(w_new * 0.6)
            # y_adj = int(h_new * 0.8)

            # we can add slight offset here, but both x_adj sum must = 2
            x11 = x_m - x_adj
            x22 = x_m + x_adj

            y11 = y_m - y_adj
            y22 = y_m + y_adj

        # truck and trains

        # x1_new = self.vehicle.x + self.plate.xmin * 0.4
        # y1_new = self.vehicle.y + self.plate.ymin * 0.3
        # x2_new = self.vehicle.x + self.plate.xmax * 1.3
        # y2_new = self.vehicle.y + self.plate.ymax * 1.4
        # normal
        # print("done adjs")

        cropped = image[int(y11): int(y22), int(x11): int(x22)]

        # tail_light = cropped

        # ratio-ing for lp_plus
        # if self.vehicle.vehicle_type == 0|1|3:
        #     print("ratio for moto")
        #     cch = cropped.shape[0]
        #     ccw = cropped.shape[1]
        #     factor = ccw / 14
        #     new_h = int(factor * 10)
        #     h_adj = int(0.5 * (new_h - (y2_new - y1_new)))

        #     tail_light = image[
        #                  int(y1_new) - h_adj : int(y2_new) + h_adj, int(x1_new) : int(x2_new)
        #                  ]
        # else:
        #     cch = cropped.shape[0]
        #     ccw = cropped.shape[1]
        #     factor = ccw / 14
        #     new_h = int(factor * 10)
        #     h_adj = int(0.5 * (new_h - (y2_new - y1_new)))
        #     tail_light = image[
        #                  int(y1_new) - h_adj : int(y2_new) + h_adj, int(x1_new) : int(x2_new)
        #                  ]

        cch = cropped.shape[0]
        ccw = cropped.shape[1]
        factor = ccw / 14
        new_h = int(factor * 10)
        h_adj = int(0.5 * (new_h - (y22 - y11)))

        # print(f"vehicle type as of tail light {self.vehicle.vehicle_type}")

        if self.vehicle.vehicle_type == 0 or 1 or 3:
            tail_light = image[
                         int(y11) - int(1.7 * h_adj): int(y22) + int(0.3 * h_adj), int(x11): int(x22)
                         ]
            #return tail_light

        elif self.vehicle.vehicle_type == 2:
            if is_lower:
                # print("for lowers")
                tail_light = image[
                             int(y11) - int(1.3 * h_adj): int(y22) + int(0.7 * h_adj), int(x11): int(x22)
                             ]
            else:
                # print("for mids")
                tail_light = image[
                             int(y11) - int(1.1 * h_adj): int(y22) + int(0.9 * h_adj), int(x11): int(x22)
                             ]

        else:
            tail_light = image[
                         int(y11) - int(h_adj): int(y22) + int(h_adj), int(x11): int(x22)
                         ]

        return tail_light

    def gen_lps(self, image):

        x1_new = self.vehicle.x + self.plate.xmin
        y1_new = self.vehicle.y + self.plate.ymin
        x2_new = self.vehicle.x + self.plate.xmax
        y2_new = self.vehicle.y + self.plate.ymax
        return image[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]

    def perform_image_proc(
            self,
            input_path,
            output_path,
            rejected_path,
            filename,
            cam_location_code,
            lane_number,
    ):

        process_state = True
        no_plate_state = True

        container_input_path = input_path.replace("/E:/", "")
        container_output_path = output_path.replace("/E:/", "")
        container_rejected_path = rejected_path.replace("/E:/", "")

        # print("container_input_path", container_input_path)
        # print("container_output_path", container_output_path)
        # print("container_rejected_path", container_rejected_path)

        enhance_filename = filename.replace(".jpg", "") + "_autozoom_wholebody.jpg"
        cropped_number_plate_filename = (
                filename.replace(".jpg", "") + "_autozoom_noplate.jpg"
        )

        final_json = {
            "numberPlate": None,
            "outputPath": None,
            "firstImage": None,
            "secondImage": None,
            "confidence": None,
            "incidentId": None,
            "status": None,
            "errorStep": None,
        }

        # get file from sftp
        try:
            image = self.read_image(container_input_path, filename)

        except FileNotFoundError:

            process_state = False

            final_json["errorStep"] = "Failed at download image step: file not found"
            final_json["status"] = "SFTP error"
            pass

        except OSError as e:

            process_state = False

            final_json["errorStep"] = "FTP socket is closed"
            final_json["status"] = "SFTP error"
            pass

        except Exception as e:

            process_state = False

            final_json["errorStep"] = str(traceback.format_exc())
            final_json["status"] = "SFTP error"
            pass

        # check lane type
        if process_state:
            if image.shape[0] > 3300:
                process_state = False
                final_json["errorStep"] = "Image Size exceeded, double databar"
                final_json["status"] = "Kesalahan Databar"

        # lane number validation
        if process_state:
            lane_number_image = image[:200, 2600:3300]

            lane, lane_number_confidence = self.ocr.perform_ocr(
                lane_number_image, False
            )

            extracted_lane_num = lane.translate(str.maketrans("", "", ":LANElane："))

            if extracted_lane_num != str(lane_number):
                process_state = False

                final_json[
                    "errorStep"
                ] = "Input lane and image lane mismatch. Double lane or mismatched lane"
                final_json["status"] = "Kesalahan Databar"

        # Vehicle detection
        if process_state:
            try:
                cropped, lp_minus = self.vehicle.detect(
                    image, cam_location_code, lane_number
                )

            except IndexError as e:

                process_state = False

                final_json["errorStep"] = "No such lane - " + str(e)
                final_json["status"] = "No violator detected"
                pass

            except TypeError as e:

                process_state = False

                final_json["errorStep"] = "No such lane - " + str(e)
                final_json["status"] = "No violator detected"
                pass

            except Exception as e:

                process_state = False

                final_json["errorStep"] = "Fail at object detection step - " + str(e)
                final_json["status"] = "No violator detected"
                pass

        # enhancing fullbody
        if process_state:
            try:
                #enhance_image, to_force = perform_image_enhancement(cropped)
                enhance_image, to_force = perform_abright(cropped)

                final_json["firstImage"] = enhance_filename

            except Exception as e:

                process_state = False

                final_json["errorStep"] = "Fail at image enhancement step - probably memory" + str(e)
                final_json["status"] = "unexpected error"
                pass

        # cropped plate result from yolov5
        if process_state:
            try:
                number_plate, confidence = self.plate.v5detect(lp_minus)
                # number_plate, confidence = self.plate.detect(lp_minus)

                if number_plate:
                    final_json["numberPlate"] = number_plate

                if confidence:
                    final_json["confidence"] = str(confidence)

                else:
                    # process_state = False
                    no_plate_state = False
                    final_json["errorStep"] = "v5 plate OCR returned null"
                    final_json["status"] = "No text detected"

            except Exception as e:

                process_state = False
                # no_plate_state = False

                final_json["errorStep"] = "Fail at plate detection - " + str(e)
                final_json["status"] = "No text detected"
                pass

        # if not number_plate:
        #     process_state = False

        #     final_json["errorStep"] = "Fail at plate detection's OCR"
        #     final_json["status"] = "No text detected"

        if process_state:

            try:
                # lp_plus = self.generate_tail_light_image(image)
                lp_plus = self.generate_tail_light_image(image)
                # print(f"lp plus is {lp_plus}")
                final_json["secondImage"] = cropped_number_plate_filename

                if not no_plate_state:
                    process_state = False

                # print("check if lp_plus is null")
                # print(lp_plus)

            except Exception as e:

                process_state = False
                final_json["errorStep"] = "Fail at lp_plus generation - " + str(e)
                final_json["status"] = "Unexpected Error"
                pass

        # if no_plate_state:
        #     process_state = False
        #     final_json["errorStep"] = "No plate detected, Attempted lp_plus genearation"
        #     # final_json["status"] = "Unexpected Error"
        #     pass

        # Number plate validation
        if process_state:
            try:
                number_plate, error_step, status, trigger = ocr_validation(number_plate)

                if trigger:
                    process_state = False
                    final_json["numberPlate"] = number_plate
                    final_json["errorStep"] = error_step
                    final_json["status"] = status

            except Exception as e:
                print("Rule based validation not run, probably no text detected")
                pass

        # final image writing to SFTP
        try:
            if process_state:
                final_json["status"] = "OK"
                final_json["outputPath"] = output_path
                self.write_image(enhance_image, enhance_filename, container_output_path)
                if to_force:
                    # self.write_image(
                    #     perform_image_enhancement_f(lp_plus), cropped_number_plate_filename, container_output_path
                    # )
                    self.write_image(
                        abright(lp_plus), cropped_number_plate_filename, container_output_path
                    )
                elif not to_force:
                    self.write_image(
                        lp_plus, cropped_number_plate_filename, container_output_path
                    )

            if not process_state:
                final_json["outputPath"] = rejected_path

                if final_json["firstImage"]:
                    self.write_image(
                        enhance_image, enhance_filename, container_rejected_path
                    )

                # if no_plate_state:

                if final_json["secondImage"]:
                    if to_force:
                        # self.write_image(
                        #     perform_image_enhancement_f(lp_plus), cropped_number_plate_filename, container_rejected_path
                        # )
                        self.write_image(
                            abright(lp_plus), cropped_number_plate_filename, container_rejected_path
                        )
                    elif not to_force:
                        self.write_image(
                            lp_plus, cropped_number_plate_filename, container_rejected_path
                        )


        except Exception as e:

            final_json["firstImage"] = None
            final_json["secondImage"] = None
            final_json["outputPath"] = None
            final_json["errorStep"] = "Failed at final image upload step - " + str(e)
            final_json["status"] = "SFTP error"
            pass

        return final_json
