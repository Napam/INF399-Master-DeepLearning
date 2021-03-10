BASE_IMG_NAME = nam012
BUILD_CMD = docker build --network=host
DOCKERFILE_TH = Dockerfile-th.docker
DOCKERFILE_TF = Dockerfile-tf.docker

USERNAME = $(shell whoami)
USERID = $(shell id -u)
GROUPID = $(shell id -g)

BUILD_ARGS = --build-arg user=$(USERNAME) \
             --build-arg uid=$(USERID) \
             --build-arg gid=$(GROUPID)

th:
	$(BUILD_CMD) $(BUILD_ARGS) -f $(DOCKERFILE_TH) -t $(BASE_IMG_NAME)-th .

tf:
	$(BUILD_CMD) $(BUILD_ARGS) -f $(DOCKERFILE_TF) -t $(BASE_IMG_NAME)-tf .	

clean: clean-th || clean-tf

clean-th:
	docker image rm $(BASE_IMG_NAME)-th

clean-tf:
	docker image rm $(BASE_IMG_NAME)-tf

