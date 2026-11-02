set -e
cd /workspace/dub/keysync
uv venv -p 3.11 venv
. venv/bin/activate
true
uv pip install --no-deps -r req_ks.txt
uv pip install "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3"
python -c "import torch,diffusers,transformers,face_alignment;print(torch.__version__,torch.cuda.is_available(),diffusers.__version__,transformers.__version__)"
echo BUILD_DONE
