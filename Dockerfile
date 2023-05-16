# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app files to the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Set the entrypoint command for running the Streamlit app
CMD ["streamlit", "run", "--server.port", "8501", "topic_model.py"]
