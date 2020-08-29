FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y curl wget
RUN wget https://dist.ipfs.io/go-ipfs/v0.6.0/go-ipfs_v0.6.0_linux-amd64.tar.gz
RUN tar xvfz go-ipfs_v0.6.0_linux-amd64.tar.gz
RUN mv go-ipfs/ipfs /usr/local/bin/ipfs
RUN ipfs init

RUN curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh
RUN apt-get install -y nodejs
RUN npm install -g ganache-cli truffle

COPY . /workspace
WORKDIR /workspace

RUN truffle compile

RUN pip install -r requirements.txt
RUN conda develop 2cp
