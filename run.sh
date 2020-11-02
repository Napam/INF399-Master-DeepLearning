#!/usr/bin/env bash
DOCKER_FLAGS=""
GPU="all"
ARGS=""
FRAMEWORK="th"

error() {
	echo "u do sumting wong"
}

while getopts "djpg:f:" option; do
	case $option in
		d) DOCKER_FLAGS+="-d";;
		g) GPU=${OPTARG};;
		j) ARGS+="./jbook.sh ";; 
		f) FRAMEWORK=${OPTARG};;
		p) DOCKER_FLAGS+=" --publish 5555:8888";;
		*) error; exit;;	
	esac
done

# $@ is an array or something, start at $OPTIND and rest
ARGS+=${@:$OPTIND}


docker run ${DOCKER_FLAGS} -it -v $(pwd)/volume:/app \
	--ipc=host --rm --gpus ${GPU} --name ${FRAMEWORK}-nam012-cntr nam012-${FRAMEWORK} ${ARGS}

