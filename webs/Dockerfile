FROM python:3.11-slim

WORKDIR /app

# Copy dataset và dữ liệu đã xử lý
COPY dataset /app/dataset
COPY loc /app/loc
COPY webs /app/webs

ENV PORT=3000
EXPOSE 3000

CMD ["python", "webs/server.py"]
