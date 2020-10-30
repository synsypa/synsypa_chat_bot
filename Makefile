HUB ?= synsypa
VERSION ?= latest

IMAGE ?= synsypa_chat_bot

.PHONY: build push pull deploy

build: Dockerfile
	docker build -t $(HUB)/$(IMAGE):$(VERSION) -f Dockerfile .

push:
	docker push $(HUB)/$(IMAGE):$(VERSION)

pull:
	docker pull $(HUB)/$(IMAGE):$(VERSION)

deploy:
	docker run --restart unless-stopped -d \
	-e DISCORD_SYNSYPA_TOKEN=$(DISCORD_SYNSYPA_TOKEN) \
	$(HUB)/$(IMAGE):$(VERSION)
