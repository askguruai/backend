FROM python:3.10

WORKDIR /app

COPY . .
COPY GOOGLE_DEFAULT_CREDENTIALS.json /etc/GOOGLE_DEFAULT_CREDENTIALS.json

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python", "main.py"]
