FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir numpy pandas matplotlib scipy

COPY . /app

CMD ["python3", "main.py"]