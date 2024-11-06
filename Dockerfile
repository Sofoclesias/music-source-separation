FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN apt-get update \
    xargs -a packages.txt apt-get install -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run","/src/app.py","--server.port=8501","--server.address=0.0.0.0"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "executable" ]