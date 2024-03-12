FROM nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /wdr

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs

COPY requirements.txt .

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .

RUN cd flash-attention && pip install . --no-build-isolation && \
    cd csrc/layer_norm && pip install . --no-build-isolation