#!/usr/bin/env bash
set -Eeuo pipefail

# All values can be overridden for parallel/local test instances.
CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-qdrant}"
IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
VOLUME_NAME="${QDRANT_VOLUME_NAME:-qdrant_storage}"
BIND_ADDRESS="${QDRANT_BIND_ADDRESS:-127.0.0.1}"
REST_PORT="${QDRANT_REST_PORT:-6333}"
GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

MIGRATION_DIR=""
BACKUP_CONTAINER=""

cleanup() {
    if [[ -n "$MIGRATION_DIR" && -d "$MIGRATION_DIR" ]]; then
        rm -rf -- "$MIGRATION_DIR"
    fi
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not on PATH." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required for the Qdrant health check." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running. Start Docker Desktop first." >&2
    exit 1
fi

# Pull before touching the running container. A failed pull must not cause downtime.
echo "Pulling ${IMAGE} ..."
docker pull "$IMAGE"

VOLUME_EXISTED=false
if docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
    VOLUME_EXISTED=true
else
    echo "Creating persistent volume ${VOLUME_NAME} ..."
    docker volume create "$VOLUME_NAME" >/dev/null
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    STORAGE_MOUNT="$(docker inspect --format \
        '{{range .Mounts}}{{if eq .Destination "/qdrant/storage"}}{{.Type}}|{{.Name}}|{{.Source}}{{end}}{{end}}' \
        "$CONTAINER_NAME")"

    if [[ "$STORAGE_MOUNT" == "volume|${VOLUME_NAME}|"* ]]; then
        echo "Replacing ${CONTAINER_NAME}; data remains in volume ${VOLUME_NAME} ..."
        docker rm --force "$CONTAINER_NAME" >/dev/null
    elif [[ -n "$STORAGE_MOUNT" ]]; then
        echo "Error: ${CONTAINER_NAME} already uses a different persistent mount:" >&2
        echo "  ${STORAGE_MOUNT}" >&2
        echo "Refusing to replace it because that could disconnect its existing data." >&2
        exit 1
    elif [[ "$VOLUME_EXISTED" == true ]]; then
        echo "Error: ${CONTAINER_NAME} stores data inside the container, but volume" >&2
        echo "${VOLUME_NAME} already exists. Refusing to merge two ambiguous data sets." >&2
        echo "Choose the data to keep, or rerun with a new QDRANT_VOLUME_NAME." >&2
        exit 1
    else
        # One-time migration from the old script, which stored data in the
        # container's writable layer. Keep that container as a recovery backup.
        BACKUP_CONTAINER="${CONTAINER_NAME}-pre-volume-$(date +%Y%m%d%H%M%S)"
        MIGRATION_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qdrant-migration.XXXXXX")"

        echo "Migrating existing container data into ${VOLUME_NAME} ..."
        docker stop "$CONTAINER_NAME" >/dev/null
        docker rename "$CONTAINER_NAME" "$BACKUP_CONTAINER"
        docker cp "${BACKUP_CONTAINER}:/qdrant/storage/." "$MIGRATION_DIR/"
    fi
fi

echo "Creating ${CONTAINER_NAME} with persistent storage ..."
docker create \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --publish "${BIND_ADDRESS}:${REST_PORT}:6333" \
    --publish "${BIND_ADDRESS}:${GRPC_PORT}:6334" \
    --volume "${VOLUME_NAME}:/qdrant/storage" \
    "$IMAGE" >/dev/null

if [[ -n "$MIGRATION_DIR" ]]; then
    docker cp "${MIGRATION_DIR}/." "${CONTAINER_NAME}:/qdrant/storage"
fi

docker start "$CONTAINER_NAME" >/dev/null

echo "Waiting for Qdrant to become healthy ..."
HEALTHY=false
for _ in {1..30}; do
    if curl --fail --silent --max-time 2 \
        "http://${BIND_ADDRESS}:${REST_PORT}/healthz" >/dev/null; then
        HEALTHY=true
        break
    fi
    sleep 1
done

if [[ "$HEALTHY" != true ]]; then
    echo "Error: Qdrant did not become healthy within 30 seconds." >&2
    docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
    exit 1
fi

echo
echo "Qdrant is healthy at http://${BIND_ADDRESS}:${REST_PORT}"
echo "Persistent volume: ${VOLUME_NAME} -> /qdrant/storage"
echo "Future image/container replacements will preserve this volume."

if [[ -n "$BACKUP_CONTAINER" ]]; then
    echo
    echo "Migration completed. Recovery container retained: ${BACKUP_CONTAINER}"
    echo "After verifying your collections, remove it with:"
    echo "  docker rm ${BACKUP_CONTAINER}"
fi
