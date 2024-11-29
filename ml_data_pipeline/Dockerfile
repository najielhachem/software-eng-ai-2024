# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install pipx and Poetry in a single layer
RUN pip install --no-cache-dir pipx && \
    pipx install poetry && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy project files
COPY . /app

# Install dependencies with Poetry
RUN poetry install

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn", "src.ml_data_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
