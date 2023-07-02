FROM ubuntu:latest

USER root

RUN apt-get update
RUN apt-get install python3 python-is-python3 pip  -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY --chown=user . .

EXPOSE 80

CMD streamlit run --server.port 80 app.py --server.enableXsrfProtection=false
