#!/bin/sh
set -e

# Fix vector_db ownership if the Docker volume was created as root.
chown -R bot:bot /app/vector_db

# Drop privileges and exec the CMD as the bot user.
exec gosu bot "$@"
