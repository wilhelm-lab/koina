FROM nvcr.io/nvidia/tritonserver:23.05-py3 AS serving-develop
RUN pip install requests ms2pip psm-utils pandas
HEALTHCHECK --start-period=1m --interval=15s --retries=12 CMD curl localhost:8501/v2/health/ready
CMD [ "/models/start.py" ]

FROM serving-develop AS serving-prod
COPY ./models  /models

FROM nvcr.io/nvidia/tritonserver:22.09-py3-sdk AS util
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get update
RUN apt-get install -y git vim curl ripgrep zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev libncursesw5 libffi-dev libreadline-dev locales
RUN locale-gen en_US.UTF-8
RUN pip install -U pip  nox poetry nox-poetry packaging
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
RUN chmod 777 /home/devuser/.pyenv/shims
# Setup node with nvm
ARG NVM_DIR="/home/devuser/.nvm"
RUN git clone https://github.com/nvm-sh/nvm.git "$NVM_DIR"
RUN cd "$NVM_DIR" && git checkout `git describe --abbrev=0 --tags --match "v[0-9]*" $(git rev-list --tags --max-count=1)`
RUN sh "$NVM_DIR/nvm.sh"
RUN chmod 777 /home/devuser/.nvm
RUN echo 'export NVM_DIR="/home/devuser/.nvm"' >> /home/devuser/.bashrc
RUN echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /home/devuser/.bashrc
RUN echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> /home/devuser/.bashrc
RUN source /home/devuser/.bashrc && nvm install 21
# Setup CI scripts
COPY ./koina_test.sh /usr/local/bin/
COPY ./koina_lint.sh /usr/local/bin/
COPY ./koina_format.sh /usr/local/bin/
USER devuser
