#!/usr/bin/env bash

docker run -it -v $(pwd)/volume:/app -p 5555:8888 --rm --gpus all nam012-${1:-th}
