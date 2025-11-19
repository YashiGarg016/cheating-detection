FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Expose Render’s dynamic port
EXPOSE 10000

# Use Render's PORT env variable instead of hardcoding 8080
CMD ["bash", "-c", "streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"]