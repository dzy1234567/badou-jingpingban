import json
import torch
from config import BASE_DIR
def dictToJson(dic, path):
    path = BASE_DIR + "\\data\\" + path
    stream = open(path, "w", encoding="utf8")
    stream.write(json.dumps(dic, ensure_ascii=False, indent=2))
    stream.close()
    return
def readFromJson(path):
    path = BASE_DIR + "\\data\\" + path
    return json.load(open(path, "r", encoding="utf8"))
def pthSave(model, path):
    path =BASE_DIR + "\\pth\\" + path
    torch.save(model.state_dict(), path)
    return
def pthLoad(model, path):
    path = BASE_DIR + "\\pth\\" + path
    model.load_state_dict(torch.load(path))
    return model
if __name__ == "__main__":
    path = "vocab3.json"
    print(BASE_DIR + "\\data\\" + path)
    dictToJson({1:20}, "vocab_test.json")
    print(readFromJson("vocab_test.json"))
    model = torch.nn.Module()
    print(model.state_dict())
    pthSave(model, "model_test.pth")
    model2 = pthLoad(model, "model_test.pth")
    print(model2.state_dict())