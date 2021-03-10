#!/usr/bin/env bash
DOCKER_FLAGS=""
ARGS=""
FRAMEWORK="th"
USER=$(whoami)
HOSTNAME="TORCH-CONTAINER"

error() {
    echo "u do sumting wong"
}

while getopts "df:g:ijpu:*" option; do
    case $option in
        d) DOCKER_FLAGS+="-d ";;
        f) FRAMEWORK=${OPTARG};;
        g) DOCKER_FLAGS+="--gpus ${OPTARG} ";;
        i) DOCKER_FLAGS+="--net host ";;
        j) ARGS+="./jbook.sh ";; 
        p) DOCKER_FLAGS+="--publish 5555:8888 ";;
        u) USER=${OPTARG};;
        *) error; exit;;	
    esac
done

# $@ is an array or something, start at $OPTIND and rest
ARGS+=${@:$OPTIND}

docker run ${DOCKER_FLAGS} -it --hostname ${HOSTNAME} \
    --user ${USER} \
    -v "$(pwd)/volume":/app \
    -v "/data/nam012/Blender/volume":/mnt/blendervol \
    --rm --name ${FRAMEWORK}-nam012-cntr nam012-${FRAMEWORK} ${ARGS}

