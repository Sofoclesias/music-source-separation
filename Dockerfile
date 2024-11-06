FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN apt-get update
RUN apt-get install --yes sox ffmpeg 
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "executable" ]