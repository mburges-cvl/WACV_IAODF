conda create --name image_annotator python=3.11.8
conda activate image_annotator
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install matplotlib pandas PyQT5 pillow numpy==1.* tqdm