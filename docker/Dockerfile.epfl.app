FROM registry.rcp.epfl.ch/dcl-zihong/qi-base:latest

# EPFL UID/GID mapping
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Copy code
COPY . /home/${LDAP_USERNAME}/qi/
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

WORKDIR /home/${LDAP_USERNAME}/qi

# Install the package
RUN pip install -e .

ENV DIFFSYNTH_MODEL_BASE_PATH=/home/${LDAP_USERNAME}/qi/checkpoints

USER ${LDAP_USERNAME}