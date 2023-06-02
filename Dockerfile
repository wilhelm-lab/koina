FROM nvcr.io/nvidia/tritonserver:23.05-py3 AS develop
RUN pip install ms2pip psm-utils pandas
HEALTHCHECK --start-period=1m --interval=15s --retries=12 CMD curl localhost:8501/v2/health/ready 

FROM develop AS prod
ADD ./models  /models
CMD [ "/models/start_triton_server.sh" ]

FROM nvcr.io/nvidia/tritonserver:22.09-py3-sdk AS util
RUN pip install -U pip pytest pylint tritonclient[all] requests black jupyter ms2pip psm-utils pandas jinja2 PyYAML
RUN apt-get update
RUN apt-get install git vim curl ripgrep -y
RUN echo '#!/bin/bash\npylint --recursive=y test models $@' > /usr/local/bin/lint
RUN chmod +x /usr/local/bin/lint
ARG UID=1000
ARG GID=1000 
RUN groupadd -f -g $GID devuser
RUN useradd -ms /bin/bash devuser -u $UID -g $GID --non-unique
USER devuser

FROM node:latest as web
ARG UID=1000
ARG GID=1000 
RUN groupadd -f -g $GID devuser
RUN useradd -ms /bin/bash devuser -u $UID -g $GID --non-unique
USER devuser