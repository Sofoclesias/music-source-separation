# Build inicial y para entrenamiento
FROM ubuntu:latest as build
WORKDIR /app/
COPY requirements.txt .
COPY /.github/ .
COPY /audiomancy/ .
RUN apt-get update
RUN apt-get install --yes sox ffmpeg python3 python3-pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash"]

# Imagen para Streamlit
FROM python:3.11-slim as stream
WORKDIR /app/
COPY --from=build /app/ .
RUN rm -rf /app/audiomancy/common/stems/
COPY /src/ .
COPY streamlit_app.py .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "executable" ]