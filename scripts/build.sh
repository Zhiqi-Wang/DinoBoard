#!/bin/bash
cd ~/projects/DinoBoard
ORT_ROOT=/root/projects/DinoBoard/third_party/onnxruntime-linux-x64-1.17.3
CFLAGS="-I$ORT_ROOT/include" LDFLAGS="-L$ORT_ROOT/lib" LD_LIBRARY_PATH="$ORT_ROOT/lib:$LD_LIBRARY_PATH" .venv_debug_shared/bin/pip install -e games/splendor/debug_service/
