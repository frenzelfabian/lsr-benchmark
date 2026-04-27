FROM ubuntu:jammy

RUN apt-get update && apt-get install -y \
    git \
    openjdk-11-jdk \
    python3 \
    python3-pip \
    build-essential \
    patchelf \
    cmake \
    onetbb-dev \
 && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /tmp/requirements.txt

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV TBB_DIR=/usr/lib/cmake/TBB
ENV CMAKE_PREFIX_PATH=/usr/lib/cmake/TBB

RUN pip3 install -r /tmp/requirements.txt

RUN python3 -c "import pyterrier as pt; pt.init();" \
 && apt-get update \
 && apt-get install -y locales \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

ADD run-pyterrier-splade.py /run-pyterrier-splade.py