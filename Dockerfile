FROM python:3.10.0rc2

EXPOSE 3000

WORKDIR /app

COPY requirements.txt .

RUN  pip install --no-cache-dir -r requirements.txt

COPY hello_microService.py /app/hello.py

CMD ["python", "hello.py"]