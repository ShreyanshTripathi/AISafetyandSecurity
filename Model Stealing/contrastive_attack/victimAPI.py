import requests
import json
import base64
import io


class VictimAPI:
    def __init__(
        self,
        port: int,
        token: int = None,
    ):
        self.port = port
        self.token = token

    def query_victim_api(self, images):
        # function to query the victim API with a list of PIL images
        endpoint = "/query"
        url = f"http://34.122.51.94:{self.port}" + endpoint
        image_data = []
        for img in images:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            image_data.append(img_base64)

        payload = json.dumps(image_data)
        response = requests.get(
            url, files={"file": payload}, headers={"token": self.token}
        )
        if response.status_code == 200:
            representation = response.json()["representations"]
            return representation
        else:
            raise Exception(
                f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
            )
