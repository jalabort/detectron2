IMAGE_NAME=ocean-eye
CONTAINER_NAME=billie
DOCKERFILE=docker/Dockerfile

build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

clean-build:
	docker build --no-cache -t $(IMAGE_NAME) -f $(DOCKERFILE) .
	
run:
	@echo "IT'S DANGEROUS TO GO ALONE! TAKE THIS."
	docker run --runtime=nvidia \
		-it --rm \
		--name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace \
		-p 8888:8888 \
		$(IMAGE_NAME)

bash:
	docker exec -it $(CONTAINER_NAME) /bin/bash
