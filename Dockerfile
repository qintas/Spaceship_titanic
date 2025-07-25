# Base image with CUDA 12.0 and cuDNN support
FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    libboost-all-dev \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip manually (latest)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install Python libraries
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    flaml \
    xgboost \
    catboost \
    shap \
    phik \
    optuna \
    ipywidgets \
    jupyterlab \
    ipykernel \
    ipython \
    statsmodels

# Expose Jupyter port
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Copy source files into container (update path if needed)
COPY . /workspace

# Start JupyterLab by default
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--ServerApp.allow_remote_access=True", "--NotebookApp.disable_check_xsrf=True"]