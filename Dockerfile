FROM python:3.11-slim

WORKDIR /app

# ✅ Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1

# ✅ Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ✅ Copy app files
COPY . .

# ✅ Expose Render’s dynamic port
EXPOSE 10000

# ✅ Use Render's PORT env variable
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "streamlit_app.py"]