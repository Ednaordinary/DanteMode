#!/bin/sh
pwd
docker run --rm -v "$(pwd):/ne/input" -it alexjc/neural-enhance --zoom=4 --type=photo /ne/input/$1.png
