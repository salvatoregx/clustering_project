run-dashboard:
	podman-compose up --build streamlit_app

run-tracking:
	podman-compose up --build mlflow

run-platform:
	podman-compose up --build --detach

stop:
	podman-compose down
