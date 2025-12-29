run-datagen:
	podman-compose up --build data_gen

run-etl:
	podman-compose up --build etl

run-model-training:
	podman-compose up --build model_training

run-dashboard:
	podman-compose up --build streamlit_app

run-pipeline:
	podman-compose up --build data_gen etl model_training

run-tracking:
	podman-compose up --build mlflow

run-all:
	podman-compose up --build --detach

stop:
	podman-compose down
