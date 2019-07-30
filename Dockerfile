FROM

ENV APP_HOME /code
WORKDIR ${APP_HOME}
ENV PATH=${PATH}:${APP_HOME}

COPY requirements.txt ${APP_HOME}
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /code/requirements.txt

RUN apt-get update && apt-get install -y iputils-ping

# Add non-root user and fix permissions
RUN groupadd --gid 716 docker && adduser --uid 11472 --gid 716 --disabled-password --quiet --gecos "" docker_user
RUN chown -Rf 11472:716 ${APP_HOME}
USER docker_user
COPY src/ ${APP_HOME}
WORKDIR ${APP_HOME}
CMD ["python", "/code/train.py"]