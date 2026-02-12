FROM ubuntu:24.04

RUN <<EOF
    set -ex

    apt update

    apt install --yes --no-install-recommends jq curl ca-certificates

    curl -sSL https://get.docker.com/ | sh

    apt clean && rm -rf /var/lib/apt/lists/*
EOF
