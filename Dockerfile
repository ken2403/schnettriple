FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update -y && apt-get install -y \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libxrender-dev \
    sudo \
    wget \
    vim \
    git \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /opt
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh && \
    sh /opt/Anaconda3-5.3.1-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm -f Anaconda3-5.3.1-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH


RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==1.9.1+cpu \
    torchvision==0.10.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    pip install --no-cache-dir schnetpack


WORKDIR /home
RUN mkdir work


CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]