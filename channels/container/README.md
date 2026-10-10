# DeepBOM container channel

The image installs one exact published DeepBOM version and runs as the
unprivileged `node` user. Model bytes remain inside the mounted local
filesystem unless the caller explicitly selects a remote input command.

```bash
docker build --build-arg DEEPBOM_VERSION=2.3.0 -t deepbom:2.3.0 -f channels/container/Dockerfile .
docker run --rm -v "$PWD:/workspace:ro" deepbom:2.3.0 audit model.onnx --summary
```

The tag is only a convenience label. Release automation must record the image
digest and the DeepBOM version separately.
