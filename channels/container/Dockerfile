ARG DEEPBOM_VERSION=1.102.0
FROM node:24.12.0-bookworm-slim@sha256:7326fb2dbdce998edd72140946851be64ef4a643e8715e138ca467e8e9d92c99
ARG DEEPBOM_VERSION
RUN npm install --global --omit=dev "deepbom@${DEEPBOM_VERSION}" \
    && npm cache clean --force
USER node
WORKDIR /workspace
ENTRYPOINT ["deepbom"]
CMD ["--help"]
