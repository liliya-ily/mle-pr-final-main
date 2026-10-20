# services/test_load.py

import os
import time
import requests


ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:4601/recommend")


def post(payload: dict) -> int:
    r = requests.post(ML_SERVICE_URL, json=payload, timeout=5)
    return r.status_code


def main() -> None:
    ok_payload = {"user_id": 15889, "month": "2016-01", "k": 5}
    bad_payload = {"user_id": "abc", "month": "2016-01", "k": 5}

    print("Smoke test: OK request")
    code = post(ok_payload)
    print("status:", code)

    print("Smoke test: validation error request")
    code = post(bad_payload)
    print("status:", code)

    print("Small load: 20 OK requests")
    for i in range(20):
        code = post(ok_payload)
        if code != 200:
            print("unexpected status:", code)
            break
        time.sleep(0.1)

    print("Done")


if __name__ == "__main__":
    main()
