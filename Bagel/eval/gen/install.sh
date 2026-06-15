
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

# sudo pip install -U openmim
# sudo mim install mmengine mmcv-full==1.7.2

# sudo pip install -U openmim
pip install mmengine mmcv-full==1.7.2


CODE_DIR=./
if [ ! -d "$CODE_DIR/mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection; git checkout 2.x
else
    cd mmdetection
fi
pip install -v -e .


cd ./eval/gen/geneval
mkdir model

bash ./evaluation/download_models.sh ./model