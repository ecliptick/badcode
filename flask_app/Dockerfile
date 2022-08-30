FROM python:3.7.12

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# layer caching for faster builds
COPY requirements.txt /
RUN pip install -r /requirements.txt

#COPY app.py /app.py
ADD . /flask_app
WORKDIR /flask_app

ENV FLASK_ENV=development

CMD flask run --host=0.0.0.0

#CMD gunicorn --workers $WORKERS \
#  --threads $THREADS \
#  --bind 0.0.0.0:$PORT_APP \
#  --log-level DEBUG \
#  app:app