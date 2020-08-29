FROM pytorch/pytorch

COPY . /workspace
WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y nodejs npm wget
RUN npm install -g npm@latest
RUN npm -v
# RUN npm install -g ganache-cli truffle

# RUN wget https://dist.ipfs.io/go-ipfs/v0.6.0/go-ipfs_v0.6.0_linux-amd64.tar.gz
# RUN tar xvfz go-ipfs_v0.6.0_linux-amd64.tar.gz
# RUN mv go-ipfs/ipfs /usr/local/bin/ipfs
# RUN ipfs init

# RUN pip install -r requirements.txt
# RUN conda develop 2cp
