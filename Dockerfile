FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install mlflow

# /home/jovyan → Default working directory in jupyter/scipy-notebook
# jovyan → user (default non-root user)

# Sets the current user as root to priveleged operations (apt-get install)
USER root
RUN apt-get update && apt-get install -y jq
# jq → a lightweight command-line JSON processor

RUN mkdir model raw_data processed_data 

ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_FILE=heart.csv


COPY heart.csv ./raw_data/heart.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py