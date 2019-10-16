FROM nvidia/cuda:10.1-cudnn7-devel

# Polyaxon Defaults
ARG local_build_dir='/build'
ARG pypi_path='54.161.171.57'
ARG quilt_path='52.70.18.116'

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
    rm -rf /var/lib/apt/lists/* && \ 
    curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Install detectron2 pypi dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision cython 'git+https://github.com/facebookresearch/fvcore' && \
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Point pip to our pypi (inside the polyaxon VPC)
# TODO[basye]: Figure out why 'https' & the domain name aren't working
RUN mkdir -p /root/.pip/ && printf "\
[global]\n\
extra-index-url = http://${pypi_path}/simple/\n\
trusted-host = ${pypi_path}\n\
" >> /root/.pip/pip.conf

# Point to our Quilt
ENV QUILT_PRIMARY_PACKAGE_DIR /data/quilt_packages
ENV QUILT_BASE /root/.local/share/QuiltCli
RUN mkdir -p $QUILT_BASE && echo "{\"registry_url\": \"http://${quilt_path}\"}" > $QUILT_BASE/config.json

# Install jupyter
RUN pip3 install --upgrade pip && pip3 install jupyter jupyterlab

WORKDIR /code

# Install Flux dependencies
COPY .${local_build_dir}/projects/Flux/requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

# Pull in detectron code
COPY .${local_build_dir} /code
RUN pip3 install .

