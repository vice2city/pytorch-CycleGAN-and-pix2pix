FROM condaforge/miniforge3:latest

WORKDIR /app/pytorch-CycleGAN-and-pix2pix
COPY environment.yml .
RUN mamba env create -f environment.yml && \
    mamba clean --all -f -y

SHELL ["/bin/bash", "--login", "-c"]
RUN mamba shell init --shell bash --root-prefix=~/.local/share/mamba
RUN echo "mamba activate opt2sar" >> ~/.bashrc
ENV PATH=/opt/conda/envs/opt2sar/bin:$PATH
ENV MAMBA_DEFAULT_ENV=opt2sar
