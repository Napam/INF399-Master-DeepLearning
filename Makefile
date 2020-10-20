BASE_IMG_NAME = nam012
BUILD_CMD = docker build --network=host
DOCKERFILE_TH = Dockerfile-PyTorch
DOCKERFILE_TF = Dockerfile-TensorFlow

th:
	$(BUILD_CMD) -f $(DOCKERFILE_TH) -t $(BASE_IMG_NAME)-th .

tf:
	$(BUILD_CMD) -f $(DOCKERFILE_TF) -t $(BASE_IMG_NAME)-tf .	

clean: clean-th || clean-tf

clean-th:
	docker image rm $(BASE_IMG_NAME)-th

clean-tf:
	docker image rm $(BASE_IMG_NAME)-tf

