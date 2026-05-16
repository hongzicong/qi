FROM ...

# Copy code
COPY . /home/${LDAP_USERNAME}/qi/
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

WORKDIR /home/${LDAP_USERNAME}/qi

# Install the package
RUN pip install -e .