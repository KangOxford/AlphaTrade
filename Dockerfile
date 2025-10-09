FROM nvcr.io/nvidia/jax:25.01-py3

# Create user
ARG UID
ARG MYUSER
RUN useradd -m -u $UID --create-home ${MYUSER} && \
    echo "${MYUSER}: ${MYUSER}" | chpasswd && \
    adduser ${MYUSER} sudo && \
    # mkdir -p /home/${MYUSER}/.local/bin && \
    chown -R  ${MYUSER}:${MYUSER} /home/${MYUSER}
    
# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=777 . .

# install tmux
RUN apt-get update && \
    apt-get install -y tmux && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libgl1 \
    p7zip-full \
    unrar \
    htop \
    graphviz \
    libcupti-dev
# Copy requirements.txt and verify its contents
COPY --chown=${MYUSER}:${MYUSER} requirements.txt /home/${MYUSER}/AlphaTrade/
RUN ls -l && cat requirements.txt

#jaxmarl from source if needed, all the requirements
# RUN pip install -e .[algs,dev]
RUN pip install --no-cache-dir -r requirements.txt


RUN wget https://go.dev/dl/go1.24.5.linux-amd64.tar.gz
RUN rm -rf /usr/local/go
RUN tar -C /usr/local -xzf go1.24.5.linux-amd64.tar.gz

RUN export PATH=$PATH:/usr/local/go/bin

USER ${MYUSER}

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Add .local/bin to PATH
RUN echo 'export PATH=$PATH:/home/duser/.local/bin' >> ~/.bashrc


# Ensure home directory is on the Python path
ENV PYTHONPATH="/home/${MYUSER}:$PYTHONPATH"
RUN export PATH="$HOME/.local/bin:$PATH"
RUN export PATH="$HOME/.local/lib:$PATH"
# Uncomment below if you want jupyter 
# RUN pip install jupyterlab

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/${MYUSER}

