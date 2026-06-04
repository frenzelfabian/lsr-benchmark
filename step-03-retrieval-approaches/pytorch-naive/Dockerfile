# amd64 keeps the CUDA-enabled PyTorch base so real submissions can use a GPU.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base-amd64

# arm64 has no CUDA PyTorch image; use a slim CPU base. CPU torch is installed
# from requirements.txt below (see --extra-index-url .../whl/cpu).
FROM python:3.11-slim AS base-arm64

# TARGETARCH is provided automatically by BuildKit (amd64 / arm64); only the
# matching stage above is built, so the CUDA image is never pulled on ARM.
FROM base-${TARGETARCH}

ADD requirements.txt /tmp/

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/* \
	&& python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

ADD search-pytorch-index.py /index-and-retrieve.py
