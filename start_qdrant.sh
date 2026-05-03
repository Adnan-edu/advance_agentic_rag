#!/bin/bash

# Stop and remove any existing Qdrant containers
EXISTING_CONTAINERS=$(docker ps -aq --filter "ancestor=qdrant/qdrant")

if [ -n "$EXISTING_CONTAINERS" ]; then
    echo "Stopping and removing existing Qdrant containers..."
    docker stop $EXISTING_CONTAINERS >/dev/null 2>&1
    docker rm $EXISTING_CONTAINERS >/dev/null 2>&1
fi

# Remove the Qdrant image to ensure a fresh pull
EXISTING_IMAGE=$(docker images -q qdrant/qdrant)

if [ -n "$EXISTING_IMAGE" ]; then
    echo "Removing existing Qdrant image..."
    docker rmi $EXISTING_IMAGE >/dev/null 2>&1
fi

# Pull the latest Qdrant image
echo "Pulling latest Qdrant image..."
docker pull qdrant/qdrant

# Run the Qdrant container with correct port mappings
echo "Starting Qdrant container..."
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    qdrant/qdrant

echo "Qdrant is now running at http://localhost:6333"
