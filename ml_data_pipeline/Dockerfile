# Use an official Python runtime as the base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Poetry and dependencies
RUN pip install pipx && pipx install poetry
RUN poetry install

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn", "src.ml_data_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
