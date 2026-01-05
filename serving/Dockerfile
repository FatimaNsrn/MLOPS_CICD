<<<<<<< HEAD
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8000"]
=======
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> 1dbc0d8a6d9c323244935092c7306f6e3f800612
