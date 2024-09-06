FROM python:3.11.5-bookworm

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
