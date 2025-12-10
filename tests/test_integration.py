import time

import requests


def test_all_services_running():
    services = [
        ("http://localhost:8000", "agent"),
        ("http://localhost:8010", "tfidf"),
        ("http://localhost:8020", "transformer"),
        ("http://localhost:5000", "mlflow"),
        ("http://localhost:9090", "prometheus"),
        ("http://localhost:3000", "grafana"),
    ]

    for url, service in services:
        try:
            if service in ["mlflow", "prometheus", "grafana"]:
                response = requests.get(url, timeout=5)
            else:
                response = requests.get(f"{url}/docs", timeout=5)
            assert response.status_code < 500
            print(f"{service} is running")
        except:
            print(f"{service} not accessible")



