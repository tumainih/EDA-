# Use slim Python 3.9 image for portability and efficiency
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (no-cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 8501 for Streamlit globally
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
