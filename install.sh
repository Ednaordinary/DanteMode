#!/bin/sh
read -rsp $'Press any key to verify you are running this script in the same location that you cloned this repo to (you should be in /DanteMode/)\n' -n1 key
echo Please install and configure ffmpeg and docker now by doing the following commands:
echo sudo apt install ffmpeg docker.io
echo sudo groupadd docker
echo sudo usermod -aG docker $USER
read -rsp $'Press any key to continue the setup\n' -n1 key
newgrp docker
python3 -m pip install -U big-sleep
python3 -m pip install -U nextcord
python3 -m pip install -U gettext
python3 -m pip install -U python-dotenv
python3 -m pip install -U shutil
python3 -m pip install -U numpy
python3 -m pip install -U torch=1.7.1
python3 -m pip install -U pillow
python3 -m pip install -U deep_daze
docker pull alexjc/neural-enhance
echo Please install CUDA now
echo https://developer.nvidia.com/cuda-downloads
echo Common mistake: if your on ubuntu, use the ubuntu install, NOT the debian one
read -rsp $'Press any key to continue the setup\n' -n1 key
touch ./.env
echo "DISCORD_TOKEN=" >> ./.env
echo Add you discord token in .env so it looks like DISCORD_TOKEN=* where * is your discord token
echo You may have to enable Hidden Files in you file manager (View > Hidden Files)
read -rsp $'Press any key to finish the setup\n' -n1 key
echo You have finished setup, you should not run it again unless .env has been deleted
