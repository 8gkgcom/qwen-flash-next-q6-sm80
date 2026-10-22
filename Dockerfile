# Build this parent from the locked upstream source; see docs/INSTALL.md.
ARG SM80_BASE_IMAGE=qwen-flash-sm80:0.2.0
FROM ${SM80_BASE_IMAGE}
WORKDIR /opt/q6-sm80
COPY Dockerfile dependencies.lock.json pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY config/example.json ./config/example.json
COPY licenses ./licenses
COPY scripts ./scripts
COPY provenance ./provenance
COPY runtime ./runtime
COPY src ./src
COPY benchmarks ./benchmarks
COPY tests ./tests
RUN python3 scripts/fetch_sources.py --only plugin llama --output /opt/q6-deps \
    && Q6_IMAGE_BUILD=1 python3 scripts/install_overlay.py --kind sm80 --target /usr/local/lib/python3.12/dist-packages \
    && Q6_IMAGE_BUILD=1 python3 scripts/install_overlay.py --kind plugin --target /opt/q6-deps/plugin \
    && python3 scripts/check_runtime_fingerprints.py
ENV CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
ENV PYTHONPATH=/opt/q6-sm80/src:/opt/q6-deps/plugin:/opt/q6-deps/llama/gguf-py:/usr/local/lib/python3.12/dist-packages
RUN cd /opt/q6-deps/plugin && python3 setup.py build_ext --inplace
RUN g++ -std=c++17 -O3 -shared -fPIC src/flashnext_gguf/q8_rows.cpp -o src/flashnext_gguf/q8_rows.so \
    && python3 -m pip install --no-deps --no-build-isolation -e . \
    && python3 -m compileall -q src scripts \
    && python3 -m unittest discover -s tests -v
ENTRYPOINT ["python3", "scripts/serve.py", "--inside-container"]
CMD ["--config", "/config/local.json"]
