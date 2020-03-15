import argparse
import torch
from my_data import MyDataset, VOCAB
from my_models import MyModel0
from my_utils import pred_to_dict
import json


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--hidden-size", type=int, default=256)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)
    dataset = MyDataset(None, args.device, test_path="data/testdict.pth")

    model.load_state_dict(torch.load("Bi-LSTM_model.pth"))

    model.eval()
    with torch.no_grad():
        for k in dataset.testdict.ks():
            texttensor = dataset.get_test_data(k)

            oupt = model(texttensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)

            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

            realtext = dataset.testdict[k]
            result = pred_to_dict(realtext, pred, prob)

            with open("results/" + k + ".json", "w", encoding="utf-8") as jsonopened:
                json.dump(result, jsonopened, indent=4)

            print(k)


if __name__ == "__main__":
    test()
