start-gpu:
	docker compose -f docker-compose.yml -f docker-compose-gpu.yml up

start-cpu-dev:
	docker compose -f docker-compose.yml -f docker-compose-cpu.yml up

start-cpu:
	docker compose -f docker-compose.yml -f docker-compose-cpu.yml up -d

start-cpu-ucr:
	docker compose -f docker-compose.yml -f docker-compose-cpu-ucr.yml up -d

build:
	docker compose -f docker-compose.yml -f docker-compose-gpu.yml build

attach: 
	docker exec -it koina_1000-serving-1 /bin/bash