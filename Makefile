run-dashboard:
	podman-compose up --build -d streamlit_app

run-tracking:
	podman-compose up --build -d mlflow

run-platform:
	podman-compose up --build -d

stop:
	podman-compose down
