run-datagen:
	podman-compose up --build data_gen

run-etl:
	podman-compose up --build etl

run-model:
	podman-compose up --build model_training

run-pipeline:
	podman-compose up --build data_gen etl model_training

run-tracking:
	podman-compose up --build mlflow

run-all:
	podman-compose up --build --detach

stop:
	podman-compose down
