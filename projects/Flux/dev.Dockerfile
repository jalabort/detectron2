FROM nvidia/cuda:10.1-cudnn7-devel
# To use this Dockerfile:
# 1. `nvidia-docker build -f dev.Dockerfile -t flux_detectron2:v0 ../..`
# 2. `nvidia-docker run -d -v /home/joan/.aws/:/root/.aws/ -v /home/joan/cvdev:/root/cvdev -p 8889:8888 -p 6007:6006 -p 23:22 --name flux_detectron_joan_gpu0 --ipc=host flux_detectron2:v0`


ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Install detectron2 pypi dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision cython 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Point pypi to hudltools
RUN mkdir -p /root/.pip/ && printf "[global]\nindex-url = http://pypi.hudltools.com\ntrusted-host = pypi.hudltools.com\n" >> /root/.pip/pip.conf

# Install Flux dependencies
COPY projects/Flux/requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

# Install jupyter
RUN pip3 install --upgrade pip
RUN pip3 install jupyter
RUN pip3 install jupyter_contrib_nbextensions
COPY projects/Flux/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user

WORKDIR /

EXPOSE 8888 6006 22
CMD jupyter notebook --no-browser --ip '0.0.0.0' --allow-root
