FROM condaforge/miniforge3:latest

WORKDIR /app/20250612
COPY environment.yml .
RUN mamba env create -f environment.yml && \
    mamba clean --all -f -y

SHELL ["/bin/bash", "--login", "-c"]
RUN echo "mamba activate 20250612" >> ~/.bashrc
ENV PATH=/opt/conda/envs/20250612/bin:$PATH
ENV MAMBA_DEFAULT_ENV=20250612
