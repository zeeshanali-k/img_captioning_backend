FROM python:3.11.4-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*
# Install any needed packages specified in requirements.txt
# RUN apt-get install gcc python3-dev musl-dev linux-headers git
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install git+https://github.com/huggingface/transformers
RUN pip install torch===2.1.0+cpu torchvision===0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

ENV OMP_NUM_THREADS=1
# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application using gunicorn (assuming 'myapp' is the Django project)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "i2t_backend.wsgi:application"]
