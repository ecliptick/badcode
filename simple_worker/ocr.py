import re
import cv2
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger("ppocr")
logger.setLevel(logging.WARN)

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
class OCR:

    model = None

    def __init__(self):
        det = "models/en_PP-OCRv3_det_distill_train"
        rec = "models/en_PP-OCRv3_rec_train"
        cls = "models/ch_ppocr_mobile_v2.0_cls_train"

        self.model = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            det_model_dir=det,
            rec_model_dir=rec,
            cls_model_dir=cls,
        )

    def recogition(self, image):
        result = self.model.ocr(image, cls=True)
        return result

    def process_ocr(self, ocr_result):

        text_len = len(ocr_result)

        if text_len == 1:
            number_plate = ocr_result[0][1][0]
            confidence = ocr_result[0][1][1]

        elif text_len == 2:

            if (
                ocr_result[0][1][0].isalpha() == True
                and ocr_result[1][1][0].isnumeric() == True
            ):
                number_plate = ocr_result[0][1][0] + ocr_result[1][1][0]
                confidence = (ocr_result[0][1][1] + ocr_result[1][1][1]) / 2

            elif (
                ocr_result[0][1][0].isnumeric() == True
                and ocr_result[1][1][0].isalpha() == True
            ):
                number_plate = ocr_result[1][1][0] + ocr_result[0][1][0]
                confidence = (ocr_result[0][1][1] + ocr_result[1][1][1]) / 2

            else:
                number_plate = ocr_result[0][1][0] + ocr_result[1][1][0]
                confidence = (ocr_result[0][1][1] + ocr_result[1][1][1]) / 2

        else:
            raise Exception("No character existed")

        number_plate = re.sub("\W+", "", number_plate)

        # validate number plate
        number_plate = number_plate.upper()
        number_plate = number_plate.replace(" ", "")

        return number_plate, confidence

    def perform_ocr(self, image, image_processing=True):

        if image_processing:
            # grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0)
            # resize image
            image = cv2.resize(
                image, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC
            )
            # init clahe
            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))
            # perform clahe
            image = clahe.apply(image)

        ocr_result = self.recogition(image)

        number_plate, confidence = self.process_ocr(ocr_result)

        return number_plate, confidence
