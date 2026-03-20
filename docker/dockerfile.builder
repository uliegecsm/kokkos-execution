FROM python:3.14

RUN <<EOF
    set -ex

    pip install system-helpers typeguard

    apt-helpers install-packages --update --packages jq curl ca-certificates

    curl -sSL https://get.docker.com/ | sh

    apt-helpers clean
EOF
