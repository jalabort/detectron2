FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda.
RUN curl -o ~/miniconda.sh -O \
    https://repo.continuum.io/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install PyTorch.
RUN conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install dependencies.
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch \
                torchvision \
                cython \
                'git+https://github.com/facebookresearch/fvcore' \
                'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
                jupyter \
                jupyterlab \
                matplotlib \
                seaborn \
                tqdm \
                knockknock

# Install detectron2.
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e /detectron2

# Configure working directory.
WORKDIR /workspace
RUN chmod -R a+w /workspace
WORKDIR /workspace

# Open Jupyter server.
EXPOSE 8888
CMD jupyter notebook --no-browser --allow-root --ip '0.0.0.0' \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --port 8888 \
