FROM nvcr.io/nvidia/tritonserver:23.05-py3 AS serving-develop
RUN pip install requests ms2pip psm-utils pandas
HEALTHCHECK --start-period=1m --interval=15s --retries=12 CMD curl localhost:8501/v2/health/ready 
CMD [ "/models/start.py" ]

FROM serving-develop AS serving-prod
COPY ./models  /models

FROM nvcr.io/nvidia/tritonserver:22.09-py3-sdk AS util
RUN pip install -U pip pytest pylint tritonclient[all] requests black jupyter ms2pip psm-utils pandas jinja2 PyYAML
RUN apt-get update
RUN apt-get install git vim curl ripgrep -y
COPY ./koina_test.sh /usr/local/bin/
COPY ./koina_lint.sh /usr/local/bin/
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g $GID devuser
RUN useradd -l -ms /bin/bash devuser -u $UID -g $GID --non-unique
RUN chmod 777 /home/devuser/
USER devuser

FROM node:latest as web
ARG UID=1000
ARG GID=1000 
RUN groupadd -f -g $GID devuser
RUN useradd -ms /bin/bash devuser -u $UID -g $GID --non-unique
USER devuser
