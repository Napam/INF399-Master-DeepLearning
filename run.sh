#!/usr/bin/env bash
DOCKER_FLAGS=""
ARGS=""
FRAMEWORK="th"

error() {
	echo "u do sumting wong"
}

while getopts "djpg:f:" option; do
	case $option in
		d) DOCKER_FLAGS+="-d ";;
		f) FRAMEWORK=${OPTARG};;
		g) DOCKER_FLAGS+="--gpus ${OPTARG} ";;
		j) ARGS+="./jbook.sh ";; 
		p) DOCKER_FLAGS+="--publish 5555:8888 ";;
		*) error; exit;;	
	esac
done

# $@ is an array or something, start at $OPTIND and rest
ARGS+=${@:$OPTIND}

docker run ${DOCKER_FLAGS} -it \
    -v "$(pwd)/volume":/app  \
    -v "/data/nam012/Blender/volume/generated_data":/mnt/generated_data\
	--ipc=host --rm --name ${FRAMEWORK}-nam012-cntr nam012-${FRAMEWORK} ${ARGS}

