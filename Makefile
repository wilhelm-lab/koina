start-gpu:
	docker compose -f docker-compose.yml -f docker-compose-gpu.yml up

start-cpu:
	docker compose -f docker-compose.yml -f docker-compose-cpu.yml up

build:
	docker compose -f docker-compose.yml -f docker-compose-gpu.yml build