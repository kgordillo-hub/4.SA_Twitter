ARG FUNCTION_DIR="/function"

#FROM python:3.8
#FROM aperture147/tensorflow-non-avx:bionic-slim
FROM tensorflow/tensorflow:2.9.3-gpu

ENV TZ="America/Bogota"
ARG DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update &&\
    apt install -y python3-pip


COPY ./ /${FUNCTION_DIR}/

WORKDIR /${FUNCTION_DIR}/

RUN pip3 install -r ${FUNCTION_DIR}/requirements.txt --no-cache-dir

EXPOSE 5000

ENTRYPOINT [ "python3" ]


CMD [ "app.py" ]