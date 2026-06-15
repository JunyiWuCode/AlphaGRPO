set -x

pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

pip install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v . 
cd ..
# /usr/bin/pip3.11 install -v -e . --no-build-isolation --no-deps
# /usr/bin/python3.11 -m pip install -v -e . --no-build-isolation

if [ ! -d ./eval/gen/geneval/model ]; then
    cd ./eval/gen/geneval
    mkdir model
    bash ./evaluation/download_models.sh ./model
fi
