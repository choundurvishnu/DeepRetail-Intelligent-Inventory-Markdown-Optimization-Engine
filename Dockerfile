# 1. Use official lightweight Python image
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirement file first (for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
COPY . .

# 6. Expose default Streamlit port (8501)
EXPOSE 8501

# 7. Command to run Streamlit app (update your script name)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
