#!/usr/bin/env bash

DOCKER_FLAGS=""
DOCKER_COMMAND=""

error() {
	echo "u do sumting wong"
}

while getopts "dj" option; do
	case $option in
		d) DOCKER_FLAGS+="-d";;
		j) DOCKER_COMMAND+="./jbook.sh";; 
		*) error; exit;;	
	esac
done

ARG1=${@:$OPTIND:1}

docker run ${DOCKER_FLAGS} -it -v $(pwd)/volume:/app -p 5555:8888 \
	--ipc=host --rm --gpus all nam012-${ARG1:-th} ${DOCKER_COMMAND}

