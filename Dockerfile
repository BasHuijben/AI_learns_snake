FROM python:3.6

WORKDIR /app

COPY functions/ ./functions
COPY default.py .
COPY requirements.txt .
COPY snake_dashboard.py .

RUN mkdir output

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["python", "snake_dashboard.py"]