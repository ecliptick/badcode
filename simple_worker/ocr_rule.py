import re


def ocr_validation(number_plate):

    error_step = None
    status = None
    trigger = False
    error_message = "OCR result is not good"
    unknown_char = re.sub("[ -~]", "", number_plate)

    digit = 0
    letter = 0

    for ch in number_plate:
        if ch.isdigit():
            digit = digit + 1
        elif ch.isalpha():
            letter = letter + 1
        else:
            pass

    if digit > 4:
        error_step = "Digit characters in number plate > 4"

    if letter > 3:
        error_step = "letter characters in number plate > 3"

    if unknown_char:
        number_plate = re.sub(r"[^\x00-\x7f]", r"", number_plate)
        error_step = "Unknown character was detected"

    if number_plate[0:3].find("O") != -1:
        error_step = "O - Q confusion"

    if len(number_plate) == 7:
        if number_plate[3:6].find("I") != -1:
            error_step = "1 - I confusion in 7 digit LP"

        elif number_plate[3:6].find("l") != -1:
            error_step = "1 - L confusion in 7 digit LP"

    # if len(number_plate) == 8:
    #     if number_plate[3] == number_plate[4]:
    #         number_plate = number_plate.replace(number_plate[4], "")
    #         error_step = "8 digit repeating characters"
    #         status = "OK with Rule triggered"
    #         trigger = True

    if len(number_plate) > 8:
        error_step = "Length of number plate > 8"

    if len(number_plate) == 6:
        if number_plate[3:5].find("I") != -1:
            error_step = "1 - I confusion in 6 digit LP"

        elif number_plate[3:5].find("l") != -1:
            error_step = "1 - L confusion in 6 digit LP"

    if len(number_plate) <= 4:
        error_step = "Too short LP"

    if number_plate[0].isnumeric():
        error_step = "Start with number"

    if number_plate.isalpha():
        error_step = "Only half LP detected, alphabet"

    if number_plate.isnumeric():
        error_step = "Only half LP detected, numeric"

    if error_step:
        status = error_message
        trigger = True

    return number_plate, error_step, status, trigger
