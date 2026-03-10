python3 -m venv .venv
source .venv/bin/activate
pip3 install Cython
pip3 install torch torchvision
pip3 install git+https://github.com/KaiyangZhou/deep-person-reid.git
pip3 install gdown
pip3 install onnx
python3 scripts/export_osnet.py