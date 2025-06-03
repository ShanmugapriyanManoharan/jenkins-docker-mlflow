FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install mlflow


USER root
RUN apt-get update && apt-get install -y jq

RUN mkdir model raw_data processed_data 


ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_FILE=heart.csv


COPY heart.csv ./raw_data/heart.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py