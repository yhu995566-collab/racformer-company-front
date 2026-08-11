#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONTAINER="${SOURCE_CONTAINER:-racformer_trt85_l20}"
NEW_CONTAINER="${NEW_CONTAINER:-q1_trt85_l20}"
NEW_IMAGE="${NEW_IMAGE:-racformer-trt85-q1:trt852-20260811}"
SERVER_REPO="${SERVER_REPO:-/home/ubuntu/hyh/3DH-Query}"
CHECKPOINT_REL="outputs/3dh_query_q1/2026-08-10/15-18-38/epoch_36.pth"
CONTAINER_REPO="/workspace/3DH-Query"
CONTAINER_OUTPUT="/workspace/outputs"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi
if [[ ! -d "$SERVER_REPO/.git" ]]; then
  echo "server repository not found: $SERVER_REPO" >&2
  exit 1
fi
if [[ ! -f "$SERVER_REPO/$CHECKPOINT_REL" ]]; then
  echo "Q1 checkpoint not found: $SERVER_REPO/$CHECKPOINT_REL" >&2
  exit 1
fi
if ! docker inspect "$SOURCE_CONTAINER" >/dev/null 2>&1; then
  echo "source TensorRT 8.5 container not found: $SOURCE_CONTAINER" >&2
  exit 1
fi
if docker inspect "$NEW_CONTAINER" >/dev/null 2>&1; then
  echo "refusing to replace existing container: $NEW_CONTAINER" >&2
  exit 1
fi

mkdir -p "$SERVER_REPO/outputs/deploy_tensorrt_q1"

echo "snapshotting $SOURCE_CONTAINER as $NEW_IMAGE"
docker commit "$SOURCE_CONTAINER" "$NEW_IMAGE"

echo "creating $NEW_CONTAINER"
docker run -d \
  --name "$NEW_CONTAINER" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v "$SERVER_REPO:$CONTAINER_REPO" \
  -v "$SERVER_REPO/outputs:$CONTAINER_OUTPUT" \
  -w "$CONTAINER_REPO" \
  --entrypoint /bin/bash \
  "$NEW_IMAGE" \
  -lc 'sleep infinity'

TRT_VERSION="$(docker exec "$NEW_CONTAINER" \
  python -c 'import tensorrt as trt; print(trt.__version__)')"
if [[ "$TRT_VERSION" != "8.5.2.2" ]]; then
  echo "unexpected TensorRT version in $NEW_CONTAINER: $TRT_VERSION" >&2
  exit 1
fi

MANIFEST="$SERVER_REPO/outputs/deploy_tensorrt_q1/q1_trt85_container_manifest.txt"
{
  echo "created at: $(date --iso-8601=seconds)"
  echo "source container: $SOURCE_CONTAINER"
  echo "new image: $NEW_IMAGE"
  echo "new image id: $(docker image inspect -f '{{.Id}}' "$NEW_IMAGE")"
  echo "new container: $NEW_CONTAINER"
  echo "TensorRT version: $TRT_VERSION"
  echo "host repository: $SERVER_REPO"
  echo "container repository: $CONTAINER_REPO"
  echo "host output: $SERVER_REPO/outputs"
  echo "container output: $CONTAINER_OUTPUT"
  echo "checkpoint: $SERVER_REPO/$CHECKPOINT_REL"
  echo "checkpoint sha256: $(sha256sum "$SERVER_REPO/$CHECKPOINT_REL" | awk '{print $1}')"
  docker inspect -f \
    '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' \
    "$NEW_CONTAINER"
} > "$MANIFEST"

cat "$MANIFEST"
echo "Q1 TensorRT 8.5 container created successfully."
