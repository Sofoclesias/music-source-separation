# Build inicial
FROM python:3.11-slim as build
WORKDIR /app
COPY min_requirements.txt .
COPY /audiomancy .
RUN apt-get update
RUN apt-get install --yes sox ffmpeg
RUN pip install --no-cache-dir -r min_requirements.txt

# Imagen para entrenamiento en colab
FROM ubuntu:latest as train
WORKDIR /app
COPY --from=build /app .
CMD ["bash"]

# Imagen para Streamlit
FROM python:3.11-slim as stream
COPY --from=build . .
RUN rm -rf app/audiomancy/common/stems
WORKDIR /app
COPY /src .
COPY /.streamlit .
COPY /streamlit_app.py .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "executable" ]