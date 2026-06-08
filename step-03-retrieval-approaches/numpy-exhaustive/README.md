# Numpy Exhaustive Retrieval

Exhaustive nearest-neighbor retrieval using cosine similarity implemented with numpy only, without additional dependencies.

## Usage

```bash
python numpy_exhaustive_search.py \
    --dataset <path-to-dataset> \
    --embedding <path-to-embeddings> \
    --output <output-dir> \
    --k 1000
```

## Run Unit Tests

```bash
pip install pytest numpy
python -m pytest test_retrieval.py -v
```

## Testing the tirex-tracker fork (Raspberry Pi PMIC energy)

The Dockerfile installs a `tirex_tracker-*.whl` from the fork that adds the
Raspberry Pi PMIC energy provider. Build that wheel on the Pi and drop it into
this directory before running the retrieval command:

```bash
# in the tirex-tracker fork checkout, on the Raspberry Pi:
cmake -S c/ -B c/build/ -G "Ninja Multi-Config" \
    --preset=conf-release-full-shared-lib-static-deps
cmake --build c/build/ --config Release --target tirex_tracker -j"$(nproc)"
cp c/build/src/Release/libtirex_tracker.so \
   python/tirex_tracker/libtirex_tracker_linux_aarch64.so
# A pretend version avoids a too-low setuptools-scm version in a fork without tags:
SETUPTOOLS_SCM_PRETEND_VERSION=0.2.99 python -m build python
cp python/dist/tirex_tracker-*.whl \
   <lsr-benchmark>/step-03-retrieval-approaches/numpy-exhaustive/
```

If loading the `.so` fails with a `GLIBCXX_…`/glibc version error, build it with a
statically linked C++ runtime by adding
`-DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc"` to the `cmake -S`
configure call.
