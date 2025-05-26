FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update && \
apt-get -y upgrade

RUN apt-get -y install ffmpeg

RUN python3 -m pip install --upgrade pip
RUN apt install -y python3.10-venv

WORKDIR /app

# docker run \
# --gpus '"device=6"' \
# -d \
# -it \
# -v neural_audio_codec:/root/.neural_audio_codec \
# --name neural_audio_codec \
# --mount type=bind,source="/home/es119256/dados/repos/minimal-neural-audio-codec",target=/app/code,bind-propagation=rshared \
# --mount type=bind,source="/home/es119256/dados/datasets/vmdb",target=/app/dataset,bind-propagation=rshared \
# neural_audio_codec

# docker exec -it neural_audio_codec sh