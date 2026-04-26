FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip build-essential patchelf libtbb-dev libtbb2

RUN pip3 install cmake

ADD requirements.txt /tmp/requirements.txt

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN pip3 install -r /tmp/requirements.txt

RUN python3 -c "import pyterrier as pt; pt.init();" && apt-get install -y locales && locale-gen en_US.UTF-8

ADD run-pyterrier-splade.py /run-pyterrier-splade.py

