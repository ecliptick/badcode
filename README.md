## Latest Update
New in v1.0.8:
- Disable paddleocr verbose
- Add non-ascii character supression

New in v1.0.7:
- Fixed bug

New in v1.0.6.2:
- Change text detection filter from 0.001 to 0.004
- Fixed bug for vehicle type
- Fixed bug for negative value for during object cropping
- Fixed non-square text image


New in v1.0.6:
- Added lp_minus to handle side by side cars
- lp_minus now replaced cropped for text detection in pipeline
- Changed hardcodes for tail light crop to scaling based
- df in car.py now has confidence and vehicle classes tagging
- Improve autozoom for motor and car
- Added class to pick variable
- For motor lp_plus = lp_minus

New in v1.0.5:
- Double databar and double lane rejection module + error handling
- new pkl as of 27/4
- production codes with no return values
- xml parsing of all robot images to compare mismatch rate

New in v1.0.4:
- Remove image combination code
- Update image processing flow in anpr.py (only 1 upload step, status = not ok by default)
- robot xml parsing notebook


## Development Environment Setup

OS : Windows/linux </br>
Python Version: 3.8</br>

1. Download

    ```bash
    git clone http://10.1.2.169/root/jpj-awas-imaging-app.git -b syed/async-pipeline
    cd jpj-awas-imaging-app
    ```
2. Start docker

    ```bash
    docker-compose up
    ```
3. Stop docker

    ```bash
    docker-compose down
    ```
4. Build specific docker file

    ```bash
    dock-compose build <image name>
    ```
5. Use `black` to format your code. 

    ```bash
    pip install black
    black .
    ```
#### Extra instruction for github repository
 Create folder `artifact` in `/simple_worker` directory and copy model to that folder.
 Create folder `weights` in `/simple_worker/craft` directory and copy `craft_mlt_25k.pth` to that folder.
