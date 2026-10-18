# Mnx Hive on a free Hugging Face Space (Docker, 2 CPUs, 16 GB RAM, no card needed).
# Put this file and README.md (from this folder) in a new Space; it builds the Hive from GitHub.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# llama.cpp's server, which runs the Mnx GGUF models (one static binary)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
 && cmake --build /opt/llama.cpp/build --target llama-server -j"$(nproc)" \
 && cp /opt/llama.cpp/build/bin/llama-server /usr/local/bin/ \
 && rm -rf /opt/llama.cpp

# Spaces run as user 1000, with the app on port 7860
RUN useradd -m -u 1000 user
ENV HOME=/home/user PLAYWRIGHT_BROWSERS_PATH=/home/user/pw MNX_PORT=7860
ARG MNX_REPO=https://github.com/Tharunmaks/Mnx.git
ARG MNX_BRANCH=ccr-e9c8dcea-1dz0oe
RUN git clone --depth 1 -b "$MNX_BRANCH" "$MNX_REPO" /home/user/app
WORKDIR /home/user/app
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m playwright install --with-deps chromium \
 && chown -R user:user /home/user

USER user
EXPOSE 7860
CMD ["bash", "scripts/hf-space.sh"]
