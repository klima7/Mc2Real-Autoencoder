FROM bitnami/pytorch:2.0.1

USER root

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

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
