FROM nvcr.io/nvidia/tritonserver:23.05-py3 AS serving-develop
RUN pip install requests ms2pip psm-utils pandas
HEALTHCHECK --start-period=1m --interval=15s --retries=12 CMD curl localhost:8501/v2/health/ready
CMD [ "/models/start.py" ]

FROM serving-develop AS serving-prod
COPY ./models  /models

FROM nvcr.io/nvidia/tritonserver:22.09-py3-sdk AS util
RUN apt-get update
RUN apt-get install -y git vim curl ripgrep zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev libncursesw5 libffi-dev libreadline-dev locales
RUN locale-gen en_US.UTF-8
RUN pip install -U pip  nox poetry nox-poetry packaging
COPY ./koina_test.sh /usr/local/bin/
COPY ./koina_lint.sh /usr/local/bin/
# Setup user
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g $GID devuser
RUN useradd -l -ms /bin/bash devuser -u $UID -g $GID --non-unique
# Setup pyenv
RUN git clone https://github.com/pyenv/pyenv.git /home/devuser/.pyenv
RUN echo 'export PYENV_ROOT="/home/devuser/.pyenv"' >> /home/devuser/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/devuser/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> /home/devuser/.bashrc
RUN chmod -R 777 /home/devuser/
RUN source /home/devuser/.bashrc && pyenv install 3.8 3.9 3.10
USER devuser


FROM node:latest as web
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g $GID devuser
RUN useradd -ms /bin/bash devuser -u $UID -g $GID --non-unique
USER devuser
