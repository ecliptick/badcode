"""Vehicle Type
ratio = 251 x 183
2 : car
3 : motorcycle
5 : bus
6 : train (sometime model detect lorry as train)
7 : lorry
"""

import cv2
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import numpy as np
from PIL import Image

class Vehicle:

    x = None
    y = None
    w = None
    h = None

    image = None
    conf_threshold = None
    nms_threshold = None

    boxes = None
    classes = None
    confidences = None

    vehicle_type = None



    def __init__(self):
        # load vehicle detection model
        self.model = cv2.dnn_DetectionModel(
            "models/vehicle_model/yolov4.cfg", "models/vehicle_model/yolov4.weights"
        )
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    def inference(self):
        self.classes, self.confidences, self.boxes = self.model.detect(
            self.image, self.conf_threshold, self.nms_threshold
        )

    def preprocess(self):
        # create dataframe for list of detected vehicles
        vehicle_df = pd.DataFrame([self.classes, self.confidences, self.boxes]).T
        vehicle_df = vehicle_df[
            (vehicle_df[0] == 0)
            | (vehicle_df[0] == 1)
            | (vehicle_df[0] == 2)
            | (vehicle_df[0] == 3)
            | (vehicle_df[0] == 5)
            | (vehicle_df[0] == 6)
            | (vehicle_df[0] == 7)
            ]

        #testing without this piece of no human filter
        # vehicle_type_list = vehicle_df[0].unique()

        # if 3 in vehicle_type_list:
        #     vehicle_df = vehicle_df[vehicle_df[0] != 0]

        vehicle_df = vehicle_df.reset_index(drop=True)

        # create dataframe for motor
        #motor_df = vehicle_df[(vehicle_df[0] == 0) | (vehicle_df[0] == 1) | (vehicle_df[0] == 3)].index
        # motor_array = np.array(motor_df)

        self.classes = vehicle_df[0].to_numpy()
        self.confidences = vehicle_df[1].to_numpy()
        self.boxes = vehicle_df[2].to_numpy()

        # resize bounding box of motor
        # for index in motor_array:
        #     self.boxes[index][0] = int(self.boxes[index][0] * 0.98)
        #     self.boxes[index][1] = int(self.boxes[index][1] * 0.90)
        #     self.boxes[index][2] = int(self.boxes[index][2] * 1.23)
        #     self.boxes[index][3] = int(self.boxes[index][3] * 1.55)

    def cbox_fr_bbox(self,bbox):
        x, y, w, h = list(map(int, bbox))
        cbox = self.image[y : y + h, x : x + w]
        return cbox

    def postprocess(self):

        #print("test if ch assign contaminate")
        #print(self.x,self.y,self.w,self.h)

        cy = self.y
        cx = self.x
        ch = self.h
        cw = self.w

        lp_minus = self.cbox_fr_bbox([self.x,self.y,self.w,self.h])

        #print("after assign ch vals")
        #print(self.x,self.y,self.w,self.h)

        #for human, vert
        #pull down bbox to capture moto
        if self.vehicle_type == 0:
            #print("trigger human")

            ch = self.h + int(self.h * 0.3)

            y_adj = int(ch*0.1)

            cy = cy - y_adj
            ch = ch + 2 * y_adj

            # lp_minus = self.image[
            #            cy - int(ch * 0.1) : cy + ch + int(ch * 0.1),
            #            cx - int(cw * 0.1) : cx + cw + int(ch * 0.1),
            #            ]

            if self.w > self.h:
                #if human landscape (mat rempit), pull up bbox to capture human
                y_adj = int(ch * 0.4)

                cy = cy - y_adj
                ch = ch + y_adj

        elif self.vehicle_type == 1 or self.vehicle_type == 3:
            #print("trigger bike or moto")
            #moto crop tightly, this is general enlarge
            #lp_minus = self.cbox_fr_bbox([self.x,self.y,self.w,self.h])
            y_adj = int(ch*0.2)
            x_adj = int(cw*0.2)
            cy = cy - y_adj
            ch = ch + 2 * y_adj
            cx = cx - x_adj
            cw = cw + 2 * x_adj



            #for only moto detection (rect)
            #drag up to crop human but dont drag down to crop more road ( or only slightly )
            #enlarge w to zoom out
            if self.w > self.h * 1.23:
                #print("trigger wide moto")
                y_adj = int(ch*0.3)
                x_adj = int(cw*0.5)

                cy = cy - y_adj
                ch = ch + 1.2 * y_adj
                cx = cx - x_adj
                cw = cw + 2 * x_adj

            #for moto extreme zoom out, increase W
            elif self.h > self.w * 1.23:
                #print("trigger vert moto - with human")
                # y_adj = 0
                x_adj = int(cw * 2)
                y_adj = int(x_adj/1.4)


                cy = cy - y_adj
                ch = ch + 2 * y_adj
                cx = cx - x_adj
                cw = cw + 2 * x_adj

            else:
                #print("trigger square moto - without human")

                y_adj = int(ch * 0.4)
                x_adj = int(cw * 0.5)

                cy = cy - y_adj
                ch = ch + 1.2 * y_adj
                cx = cx - x_adj
                cw = cw + 2 * x_adj

            #cropped = self.cbox_fr_bbox([cx,cy,cw,ch])
            #pil_img = Image.fromarray(cropped)
            # width, height = pil_img.size
            # # bar = Image.new(pil_img.mode, (int(0.25 * width), height), (0, 0, 0))
            # # pil_img.paste(bar, (0, 0))



        #for car and truck
        else:
            #print("trigger normal")
            #print(cx,cy,cw,ch)
            cropped = self.cbox_fr_bbox([cx,cy,cw,ch])
            pil_img = Image.fromarray(cropped)
            width, height = pil_img.size
            bar = Image.new(pil_img.mode, (int(0.45 * width), height), (0, 0, 0))
            pil_img.paste(bar, (0, 0))
            lp_minus = np.array(pil_img)


        #print(f"bbox derived{cx,cy,cw,ch}")
        #print(f"bbox check if mutate{self.x,self.y,self.w,self.h}")
        #final zoom out if for square-ish crops (H> W/1.4

        if ch > cw / 1.4 and self.vehicle_type != 1|0|3:
            #print("triggered square")
            #print(f"bbox unedited{cx,cy,cw,ch}")

            # y_adj = int(ch * 0.1)
            y_adj = int(ch * 0.01)
            cy = cy - y_adj
            ch = ch + (2 * y_adj)

            x_adj = (ch - (cw / 1.4)) * 1.4/2 * 1

            cx = cx - x_adj
            cw = cw + (2 * x_adj)

            #print(f"bbox {cx,cy,cw,ch}")

        #add missing h pixels to fit ratio
        #ratio here
        else:
            # elif self.vehicle_type != 1|0|3:
            #print("triggered regular ratio")
            #(f"bbox unedited{cx,cy,cw,ch}")
            cropped = self.cbox_fr_bbox([cx,cy,cw,ch])
            cch = cropped.shape[0]
            ccw = cropped.shape[1]

            new_h = int(cw / 1.4)

            y_adj = (new_h - ch)/2

            cy = cy - y_adj
            ch = ch + 2 * y_adj

        cx, cy, cw, ch = int(cx), int(cy), int(cw), int(ch)

        cropped = self.cbox_fr_bbox([cx,cy,cw,ch])

        # resize_cropped = cv2.resize(
        #     cropped, (cropped.shape[1] * 3, cropped.shape[0] * 3)
        # )
        resize_cropped = cropped

        return resize_cropped, lp_minus

    def vehicle_selection(self, cam_loc, lane_num):
        # load midpoint list
        midpoint_pkl = open("mid.pkl", "rb")
        midpoint_list = pickle.load(midpoint_pkl)
        midpoint_pkl.close()

        try:
            mid_g_x, mid_g_y = literal_eval(
                np.array(
                    midpoint_list[
                        midpoint_list["Host name / Site Code"] == cam_loc
                        ].iloc[:, lane_num + 5]
                )[0]
            )

        except IndexError:
            raise Exception("no such lane at .pkl - out of df")

        except ValueError:
            raise Exception(
                "no such lane at .pkl - location has too few lanes for this input"
            )

        except Exception as e:
            raise Exception("mid point issue" + str(e))

        # find actual vehicle
        if len(self.boxes) == 1:
            pick = pd.Series(self.boxes[0])
            pick["class"] = self.classes[0]

        elif len(self.boxes) > 1:

            # create new dataframe
            df1 = pd.DataFrame(self.boxes.tolist())
            df2 = pd.DataFrame([self.classes, self.confidences]).T

            # combine 2 dataframe
            df = pd.DataFrame(pd.concat([df2, df1], axis=1))
            df.columns = ["class", "conf", "x", "y", "w", "h"]
            df["class"] = df["class"].astype("int")

            # calculate midpoint axis x
            df["mid_x"] = df["x"] + (df["w"] * 0.5)

            # calculate midpoint axis y
            df["mid_y"] = df["y"] + (df["h"] * 0.5)

            # calculate distance between preset midpoint and object midpoint using Euclidean Distance formula
            df["dist"] = (
                                 (mid_g_x - df["mid_x"].astype(int)) ** 2
                                 + (mid_g_y - df["mid_y"].astype(int)) ** 2
                         ) ** 0.5

            # sort dataframe in ascending order
            df.sort_values("dist", inplace=True)

            # reset index after sorting
            df = df.reset_index(drop=True)

            # choose the closest distance from midpoint
            pick = df.iloc[0, 2:6].astype(int)
            pick["class"] = df.iloc[:1]["class"].values[0]


        else:
            raise Exception("no car was detected")

        # crop selected vehicle
        self.x, self.y, self.w, self.h, self.vehicle_type = pick
        #print(f"picked box {pick}")

    def detect(self, image, cam_loc, lane_num):

        self.conf_threshold = 0.4
        self.nms_threshold = 0.6
        self.image = image

        self.inference()
        # print("v4 raw boxes")
        # print(self.boxes)

        if len(self.boxes) == 0:
            raise Exception("no car was detected")

        self.preprocess()

        try:
            self.vehicle_selection(cam_loc, lane_num)
        except Exception as e:
            raise Exception(str(e))

        resize_cropped, lp_minus = self.postprocess()

        return resize_cropped, lp_minus