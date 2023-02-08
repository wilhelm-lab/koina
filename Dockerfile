FROM nvcr.io/nvidia/tritonserver:22.09-py3 AS develop
RUN pip install ms2pip psm-utils pandas
HEALTHCHECK --start-period=1m CMD curl localhost:8501/v2/health/ready 

FROM develop AS prod
ADD ./models  /models
CMD [ "/models/start_triton_server.sh" ]

FROM python:3.8.16-slim AS util
RUN pip install pytest pylint tritonclient[all] requests black
RUN apt-get update
RUN apt-get install git vim -y
RUN echo '#!/bin/bash\npylint --recursive=y test models $@' > /usr/local/bin/lint
RUN chmod +x /usr/local/bin/lint
ARG UID=1001 # 1001 is the default if you build it with docker-compose it automatically sets it to your UID
ARG GID=1001 
RUN groupadd -g $GID devuser
RUN useradd -ms /bin/bash devuser -u $UID -g $GID
USER devuser
