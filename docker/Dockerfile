ARG UBUNTU_VERSION                      # Eg. 22.04
FROM ubuntu:${UBUNTU_VERSION}
# To build this image from this folder with OpenFOAM v2206:
# docker build --build-arg OPENFOAM_VERSION=2206 --build-tag SMARTSIM_VERSION=0.7.0 -t openfoam-smartsim:latest .

ARG SMARTSIM_VERSION                    # Eg. 0.6.2
ARG OPENFOAM_VERSION                    # Eg. 2206
ARG DATABASE_BACKEND=redis              # redis or keydb
LABEL maintainer="Mohammed Elwardi Fadeli <elwardi.fadeli@tu-darmstadt.de>"
LABEL source="https://github.com/OFDataCommittee/openfoam-smartsim"

# Install and setup
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/usr/bin/bash
RUN apt update -y && \
    apt install -y --no-install-recommends git-lfs libpython3-dev pip cmake \
        sudo apt-utils vim curl wget software-properties-common \
        git-core cmake openssh-server gcc g++  \
        binutils libopenmpi-dev openmpi-bin openmpi-common openmpi-doc 

# Set up OpenFOAM repositories
RUN sh -c "curl https://dl.openfoam.com/add-debian-repo.sh | bash"
RUN apt update -y && \
    apt install -y --no-install-recommends openfoam${OPENFOAM_VERSION}-default
RUN apt clean && apt purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Set up smartsim
RUN pip install smartsim==${SMARTSIM_VERSION} && \
    if [ "${DATABASE_BACKEND}" = "keydb" ]; then smart build --device cpu --no_tf --no_pt; else smart build --device cpu --no_tf --no_pt --keydb; fi
ENV USER openfoam
ENV HOME=/home/${USER} 
RUN adduser --disabled-password --gecos "" ${USER} -u 1001 && \
    echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER openfoam
RUN echo "source /usr/lib/openfoam/openfoam${OPENFOAM_VERSION}/etc/bashrc" >> ${HOME}/.bashrc
RUN . "/usr/lib/openfoam/openfoam${OPENFOAM_VERSION}/etc/bashrc" && mkdir -p $FOAM_USER_LIBBIN && mkdir -p $FOAM_USER_APPBIN


#COPY bin/uid_entrypoint /usr/local/bin/uid_entrypoint
#ENTRYPOINT ["uid_entrypoint"]
