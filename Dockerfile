FROM nvcr.io/nvidia/tritonserver:22.09-py3 AS develop
RUN pip install ms2pip psm-utils pandas

FROM develop AS prod
ADD ./models  /models
CMD [ "/models/start_triton_server.sh" ]
