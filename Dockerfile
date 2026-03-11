FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl

# Remove NGC's pip cmake wrapper (breaks during pip dependency resolution)
RUN pip uninstall -y cmake 2>/dev/null || true

# System packages for MuJoCo EGL rendering and ffmpeg build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential pkg-config nasm \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libosmesa6-dev \
    libx264-dev libx265-dev libvpx-dev \
    git wget \
    && rm -rf /var/lib/apt/lists/*

# Build SVT-AV1 (required by ffmpeg's libsvtav1 encoder)
RUN git clone --depth 1 --branch v2.3.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git /tmp/svtav1 && \
    cd /tmp/svtav1/Build && \
    cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/svtav1

# Build ffmpeg 7.1 from source with libsvtav1 support
RUN git clone --depth 1 --branch n7.1 https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
        --prefix=/usr/local \
        --enable-shared --disable-static \
        --enable-gpl \
        --enable-libsvtav1 \
        --enable-libx264 --enable-libx265 --enable-libvpx \
        --disable-doc --disable-ffplay && \
    make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /tmp/ffmpeg

# Verify ffmpeg 7.x with libsvtav1
RUN ffmpeg -version | head -1 && \
    ffmpeg -encoders 2>/dev/null | grep -q svtav1 && \
    echo "ffmpeg 7.x with libsvtav1 OK"

# Install PyAV from source against new ffmpeg, then lerobot
RUN pip install --no-cache-dir --no-binary av "av>=15.0"
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/nightly/cu126 \
    "lerobot[smolvla,libero]" "torchao>=0.17.0.dev0" && \
    python -c "import torch; assert torch.version.cuda, 'CUDA support lost after lerobot install'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "from torchao import __version__; print(f'torchao {__version__}')"

# Patch diffusers bug: logger not defined in torchao_quantizer.py
# (https://github.com/huggingface/diffusers/pull/11018)
RUN TQFILE=$(find /usr/local/lib -path "*/diffusers/quantizers/torchao/torchao_quantizer.py" | head -1) && \
    test -n "$TQFILE" && \
    sed -i '1i import logging\nlogger = logging.getLogger(__name__)' "$TQFILE" && \
    echo "Patched $TQFILE"

# Smoke-test the full import chain (diffusers → lerobot)
RUN python -c "from transformers import AutoProcessor; print('AutoProcessor import OK')" && \
    python -c "from diffusers import ConfigMixin, ModelMixin; print('diffusers import OK')"

WORKDIR /workspace
COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
