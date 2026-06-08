# Base matches the Raspberry Pi OS glibc (Debian bookworm, glibc 2.36) so that the
# prebuilt aarch64 libtirex_tracker.so from the fork loads without an ABI mismatch.
FROM python:3.11-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/*

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Override the PyPI tirex-tracker (pulled transitively by lsr-benchmark) with the fork
# build that adds the Raspberry Pi PMIC energy provider. The wheel bundles the prebuilt
# aarch64 .so (see README for how to build it).
ADD tirex_tracker-*.whl /tmp/
RUN pip3 install --no-cache-dir --force-reinstall --no-deps /tmp/tirex_tracker-*.whl

ADD numpy_exhaustive_search.py /index-and-retrieve.py
