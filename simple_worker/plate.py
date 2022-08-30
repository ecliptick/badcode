import torch
import re
import pandas as pd

from ocr import OCR

ocr = OCR()


class Plate:
    ymin = None
    ymax = None
    xmin = None
    xmax = None

    error_type = None

    plate_state = True

    def __init__(self):
        self.model = torch.hub.load(
            "components/yolov5",
            "custom",
            path="models/plate_model/yolov5.pt",
            force_reload=True,
            source="local",
        )
        self.model.conf = 0.15

    def count_alphabet(self, number_plate):

        letter_num = 0

        for ch in number_plate:
            if ch.isalpha():
                letter_num = letter_num + 1
            else:
                pass

        return letter_num

    def v5detect(self, image):

        detections = self.model(image[..., ::-1])

        results = detections.pandas().xyxy[0].to_dict(orient="records")

        num_plate = len(results)

        if num_plate == 1:
            self.ymin = results[0]["ymin"]
            self.ymax = results[0]["ymax"]
            self.xmin = results[0]["xmin"]
            self.xmax = results[0]["xmax"]

            plate_image = image[
                          int(self.ymin) : int(self.ymax), int(self.xmin) : int(self.xmax)
                          ]

            try:
                final_ocr_plate, final_confidence = ocr.perform_ocr(plate_image)

            except Exception as e:
                self.error_type = "ocr"
                print("OCR error "+str(e))
                final_ocr_plate, final_confidence = None, None

                pass

        elif num_plate > 1:
            #print("more than 1 lp")

            plate_df = pd.DataFrame(results)
            plate_df.insert(loc=1, column="status", value=True)
            plate_df.insert(loc=1, column="number_plate", value=None)
            plate_df.insert(loc=1, column="plate_confidence", value=None)

            plate_image_1 = image[
                            int(plate_df.iloc[0].ymin) : int(plate_df.iloc[0].ymax),
                            int(plate_df.iloc[0].xmin) : int(plate_df.iloc[0].xmax),
                            ]
            plate_image_2 = image[
                            int(plate_df.iloc[1].ymin) : int(plate_df.iloc[1].ymax),
                            int(plate_df.iloc[1].xmin) : int(plate_df.iloc[1].xmax),
                            ]

            try:
                ocr_plate_image_1, confidence_1 = ocr.perform_ocr(plate_image_1)
                plate_df.at[0, "number_plate"] = ocr_plate_image_1
                plate_df.at[0, "plate_confidence"] = confidence_1

            except Exception as e:
                plate_df.at[0, "status"] = False
                pass

            try:
                ocr_plate_image_2, confidence_2 = ocr.perform_ocr(plate_image_2)
                plate_df.at[1, "number_plate"] = ocr_plate_image_2
                plate_df.at[1, "plate_confidence"] = confidence_2
            except Exception as e:
                plate_df.at[1, "status"] = False
                pass

            if plate_df.at[0, "status"]:
                if "/" in ocr_plate_image_1:
                    plate_df.at[0, "status"] = False

            if plate_df.at[1, "status"]:
                if "/" in ocr_plate_image_2:
                    plate_df.at[1, "status"] = False

            if plate_df.at[0, "status"]:
                if self.count_alphabet(ocr_plate_image_1) > 4:

                    plate_df.at[0, "status"] = False

            if plate_df.at[1, "status"]:
                if self.count_alphabet(ocr_plate_image_2) > 4:

                    plate_df.at[1, "status"] = False

            filtered_df = plate_df[plate_df["status"] == True]

            filtered_df = filtered_df.reset_index(drop=True)

            detected_num_plate = filtered_df.shape[0]

            if detected_num_plate < 1:

                self.ymin = plate_df.at[0, "ymin"]
                self.ymax = plate_df.at[0, "ymax"]
                self.xmin = plate_df.at[0, "xmin"]
                self.xmax = plate_df.at[0, "xmax"]
                self.plate_state = False
                raise Exception("Zero plate was detected")

            self.ymin = filtered_df.at[0, "ymin"]
            self.ymax = filtered_df.at[0, "ymax"]
            self.xmin = filtered_df.at[0, "xmin"]
            self.xmax = filtered_df.at[0, "xmax"]

            final_ocr_plate = filtered_df.at[0, "number_plate"]
            final_confidence = filtered_df.at[0, "plate_confidence"]

        else:
            self.error_type = "plate"
            self.plate_state = False
            raise Exception("Zero plate was detected")

        if final_ocr_plate:

            final_ocr_plate = re.sub("\W+", "", final_ocr_plate)

        # else:
        #     raise Exception("No text detected at plate detection")

        #print(self.ymin,self.ymax,self.xmin,self.xmax)

        # if not final_ocr_plate:
        #     final_ocr_plate = None

        # if not final_confidence:
        #     final_confidence = None


        return final_ocr_plate, final_confidence

        # num_plate = len(results)
        #
        # print(results)
        # return results
