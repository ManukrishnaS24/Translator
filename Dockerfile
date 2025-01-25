# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the backend files (Flask app)
COPY app.py /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy static files (HTML, CSS, JavaScript, images)
COPY static/ /app/static

# Expose the port the app will run on
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]
