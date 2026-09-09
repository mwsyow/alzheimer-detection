FROM antsx/ants:v2.6.5 AS ants

FROM python:3.13-slim

ARG CLINICA_VERSION=0.11.3

COPY --from=ants /opt/ants /opt/ants

ENV PATH="/opt/ants/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/ants/lib"

RUN python -m pip install --no-cache-dir "clinica==${CLINICA_VERSION}"

# Clinica otherwise downloads this reference into its package directory at
# runtime, which is deliberately read-only when the pipeline runs as the host
# user. Clinica verifies the resource against its built-in SHA-256.
RUN python -c "from clinica.utils.image import get_mni_cropped_template, get_mni_template; print(get_mni_template('t1')); print(get_mni_cropped_template())"

ENTRYPOINT []
CMD ["clinica", "--help"]
