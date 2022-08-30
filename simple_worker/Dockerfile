FROM python:3.7.12

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# layer caching for faster builds
COPY requirements.txt /

RUN pip install -r /requirements.txt
RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
# RUN pip install celery
# RUN pip install matplotlib==3.5.1
# RUN pip install opencv-python==4.5.4.60
# RUN pip install opencv-python-headless==4.5.4.60
# RUN pip install pandas==1.3.5
# RUN pip install paramiko==2.10.3
# RUN pip install pickle-mixin==1.0.2
# RUN pip install python-bidi==0.4.2
# RUN pip install PyYAML==6.0
# RUN pip install scikit-image==0.19.2
# RUN pip install scipy==1.7.3
# RUN pip install paddleocr
# RUN pip install paddlepaddle
# RUN pip install bbox 
# RUN pip install python-dotenv

#COPY app.py /app.py
ADD . /simple_worker
WORKDIR /simple_worker

ENTRYPOINT celery -A tasks worker --loglevel=info
