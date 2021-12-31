FROM python:3.7
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV FLASK_ENV development
EXPOSE 5000
CMD ["python", "app.py"]