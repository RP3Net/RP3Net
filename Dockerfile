# docker build --platform linux/amd64 -t rp3net:test_1 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y git wget && \
    mkdir -p /rp3/python && \
    cd /rp3 && \
    python -m venv --clear --system-site-packages python && \
    /rp3/python/bin/pip install --upgrade pip && \
    /rp3/python/bin/pip install RP3Net jupyter && \
    /rp3/python/bin/jupyter server --generate-config && \
    wget -nv -nc https://ftp.ebi.ac.uk/pub/software/RP3Net/v0.1/checkpoints/rp3net_v0.1_d.ckpt && \
    wget -nv -nc https://raw.githubusercontent.com/RP3Net/RP3Net/refs/heads/main/rp3_colab.ipynb

# /root/.jupyter/jupyter_server_config.py is generarated by `jupyter server --generate-config`
RUN <<EOT cat >> /root/.jupyter/jupyter_server_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ExtensionApp.open_browser = False
c.IdentityProvider.token = ''
EOT

ENV PATH="/rp3/python/bin:$PATH"
WORKDIR /rp3
EXPOSE 8888
