from attack import APIModelStealer
from dataset import QueryDataset, TaskDataset
import requests
import torch
import sys
import pdb
import json
import pickle
import time
from class_ratio import class_ratio
from victimAPI import VictimAPI

TOKEN = "55172888"


def request_API():
    response = requests.get(
        "http://34.122.51.94:9090" + "/stealing_launch", headers={"token": TOKEN}
    )
    answer = response.json()
    print(answer)  # {"seed": "SEED", "port": PORT}
    if "detail" in answer:
        with open("creds.json", "r") as f:
            loaded_dict = json.load(f)
        SEED = str(loaded_dict["seed"])
        PORT = str(loaded_dict["port"])
        return SEED, PORT
    # save the values
    SEED = str(answer["seed"])
    PORT = str(answer["port"])
    with open("creds.json", "w") as f:
        json.dump(answer, f)
    return SEED, PORT


def save_embeddings(embeddings, ids):
    embeddings_dict = dict(zip(ids, embeddings))
    with open("embeddings.pickle", "wb") as handle:
        pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings_dict


def get_embeddings_from_API(query_dataset):
    # Get API credentials
    pdb.set_trace()
    seed, port = request_API()
    vicapi = VictimAPI(port, TOKEN)
    embeddings = []
    # Make queries to the API and collect embeddings
    num_partitions = int(len(query_dataset) / 1000)
    for i in range(num_partitions):
        print(i)
        start_ind = i * 1000
        embeddings.extend(
            vicapi.query_victim_api(
                query_dataset.selected_imgs[start_ind : start_ind + 1000]
            )
        )
        time.sleep(65)
    if len(query_dataset) % 1000 != 0:
        embeddings.extend(
            vicapi.query_victim_api(
                query_dataset.selected_imgs[(num_partitions - 1) * 1000 :]
            )
        )
    embeddings_dict = save_embeddings(embeddings, query_dataset.selected_ids)
    return embeddings_dict


def run_contrastive_model_stealing_attack():
    dataset = torch.load("../ModelStealingPub.pt", weights_only=False)
    query_dataset = QueryDataset(dataset=dataset, class_ratio=class_ratio)
    embeddings_dict = get_embeddings_from_API(query_dataset)
    with open("embeddings.pickle", "rb") as f:
        embeddings_dict = pickle.load(f)
    # Initialize the stealer
    stealer = APIModelStealer(
        surrogate_arch="resnet18",
        embedding_dim=1024,
        loss_type="infonce",  # Options: 'mse', 'infonce', 'cosine'
    )
    # pdb.set_trace()
    # Execute the attack
    stolen_model = stealer.steal_model(
        query_dataset=query_dataset,
        embeddings=embeddings_dict,
        batch_size=32,
        epochs=50,
        learning_rate=0.0003,
    )

    # Save the stolen model
    stealer.save_stolen_model("stolen_contrastive_model.pth")


if __name__ == "__main__":
    # dictionary = {"seed": 77149021, "port": "9306"}
    # with open("creds.json", "w") as f:
    #     json.dump(dictionary, f)
    run_contrastive_model_stealing_attack()
