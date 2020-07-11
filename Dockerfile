FROM pytorch/pytorch

RUN apt update
RUN apt install -y nodejs npm
RUN npm install -g ganache-cli

RUN apt install -y wget
RUN wget https://dist.ipfs.io/go-ipfs/v0.6.0/go-ipfs_v0.6.0_linux-amd64.tar.gz
RUN tar xvfz go-ipfs_v0.6.0_linux-amd64.tar.gz
RUN mv go-ipfs/ipfs /usr/local/bin/ipfs

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt
RUN conda develop client

