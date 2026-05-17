FROM qi-base:latest

# Copy code
COPY . /root/qi/
RUN chown -R root:root /root/qi

WORKDIR /root/qi

# Install the package (torch/torchvision already installed in base)
RUN pip install -e . 

USER root