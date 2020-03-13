import json
import os
import random
from os import path
from string import ascii_uppercase, digits, punctuation

import colorama
import numpy
import regex
import torch
from colorama import Fore
from torch.utils import data

from my_classes import TextBox, TextLine
from my_utils import robust_padding

VOCAB = ascii_uppercase + digits + punctuation + " \t\n"


class MyDataset(data.Dataset):
    def __init__(
        self, dictpath="data/data_dict.pth", device="cpu", val_size=76, testpath=None
    ):
        if dictpath is None:
            self.valdict = {}
            self.traindict = {}
        else:
            data_items = list(torch.load(dictpath).items())
            random.shuffle(data_items)

            self.valdict = dict(data_items[:val_size])
            self.traindict = dict(data_items[val_size:])

        if testpath is None:
            self.testdict = {}
        else:
            self.testdict = torch.load(testpath)

        self.device = device

    def get_test_data(self, key):
        text = self.testdict[key].upper()
        texttensor = torch.zeros(len(text), 1, dtype=torch.long)
        texttensor[:, 0] = torch.LongTensor([VOCAB.find(c) for c in text])

        return texttensor.to(self.device)

    def get_train_data(self, batch_size=8):
        samples = random.sample(self.traindict.keys(), batch_size)

        texts = [self.traindict[k][0] for k in samples]
        lbls = [self.traindict[k][1] for k in samples]

        robust_padding(texts, lbls)

        maxlen = max(len(t) for t in texts)

        texttensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, text in enumerate(texts):
            texttensor[:, i] = torch.LongTensor([VOCAB.find(c) for c in text])

        truthtensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, labl in enumerate(lbls):
            truthtensor[:, i] = torch.LongTensor(labl)

        return texttensor.to(self.device), truthtensor.to(self.device)

    def get_val_data(self, batch_size=8, device="cpu"):
        keys = random.sample(self.valdict.keys(), batch_size)

        texts = [self.valdict[k][0] for k in keys]
        lbls = [self.valdict[k][1] for k in keys]

        maxlen = max(len(s) for s in texts)
        texts = [s.ljust(maxlen, " ") for s in texts]
        lbls = [
            numpy.pad(a, (0, maxlen - len(a)), mode="constant", constant_values=0)
            for a in lbls
        ]

        texttensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, text in enumerate(texts):
            texttensor[:, i] = torch.LongTensor([VOCAB.find(c) for c in text])

        truthtensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, labl in enumerate(lbls):
            truthtensor[:, i] = torch.LongTensor(labl)

        return keys, texttensor.to(self.device), truthtensor.to(self.device)


def get_files(data_path="data/"):
    jsonfiles = sorted(
        (f for f in os.scandir(data_path) if f.name.endswith(".json")),
        key=lambda f: f.path,
    )
    txtfiles = sorted(
        (f for f in os.scandir(data_path) if f.name.endswith(".txt")),
        key=lambda f: f.path,
    )

    assert len(jsonfiles) == len(txtfiles)
    for f1, f2 in zip(jsonfiles, txtfiles):
        assert path.splitext(f1)[0] == path.splitext(f2)[0]

    return jsonfiles, txtfiles


def sort_text(txt_file):
    with open(txt_file, "r") as txt_opened:
        content = sorted([TextBox(line) for line in txt_opened], key=lambda box: box.y)

    textlines = [TextLine(content[0])]
    for box in content[1:]:
        try:
            textlines[-1].insert(box)
        except ValueError:
            textlines.append(TextLine(box))

    return "\n".join([str(text_line) for text_line in textlines])


def create_test_data():
    keys = sorted(
        path.splitext(f.name)[0]
        for f in os.scandir("tmp/task3-test(347p)")
        if f.name.endswith(".jpg")
    )

    files = ["tmp/text.task1&2-test(361p)/" + s + ".txt" for s in keys]

    testdict = {}
    for k, f in zip(keys, files):
        testdict[k] = sort_text(f)

    torch.save(testdict, "data/testdict.pth")


def create_data(data_path="tmp/data/"):

    jsonfiles, txtfiles = get_files(data_path)
    keys = [path.splitext(f.name)[0] for f in jsonfiles]

    data_dict = {}

    for key, json_file, txt_file in zip(keys, jsonfiles, txtfiles):
        with open(json_file, "r", encoding="utf-8") as json_opend:
            key_info = json.load(json_opend)

        text = sort_text(txt_file)
        textspace = regex.sub(r"[\t\n]", " ", text)

        textclass = numpy.zeros(len(text), dtype=int)

        print()
        print(json_file.path, txt_file.path)
        for i, k in enumerate(iter(key_info)):
            v = key_info[k]
            if k == "total":
                s = regex.search(
                    r"(\bTOTAL[^C]*ROUND[^C]*)(" + v + r")(\b)", textspace
                )
                if s is None:
                    s = regex.search(r"(\bTOTAL[^C]*)(" + v + r")(\b)", textspace)
                    if s is None:
                        s = regex.search(r"(\b)(" + v + r")(\b)", textspace)
                        if s is None:
                            s = regex.search(r"()(" + v + r")()", textspace)
                v = s[2]
                textclass[range(*s.span(2))] = i + 1
            else:
                if not v in textspace:
                    s = None
                    e = 0
                    while s is None and e < 3:
                        e += 1
                        s = regex.search(
                            r"(\b" + v + r"\b){e<=" + str(e) + r"}", textspace
                        )
                    v = s[0]

                pos = textspace.find(v)
                textclass[pos : pos + len(v)] = i + 1

        data_dict[key] = (text, textclass)

        # print(txt_file.path)
        # color_print(text, textclass)

    return keys, data_dict


def color_print(text, textclass):
    colorama.init()
    for c, n in zip(text, textclass):
        if n == 1:
            print(Fore.RED + c, end="")
        elif n == 2:
            print(Fore.GREEN + c, end="")
        elif n == 3:
            print(Fore.BLUE + c, end="")
        elif n == 4:
            print(Fore.YELLOW + c, end="")
        else:
            print(Fore.WHITE + c, end="")
    print(Fore.RESET)
    print()


if __name__ == "__main__":
    create_test_data()
    create_data()
    get_files()
    create_test_data()

    # dataset = MyDataset("data/data_dict2.pth")
    # text, truth = dataset.get_train_data()
    # print(text)
    # print(truth)
    # dict3 = torch.load("data/data_dict3.pth")
    # for k in dict3.keys():
    #     text, textclass = dict3[k]
    #     color_print(text, textclass)

    # keys, data_dict = create_data()
    # torch.save(data_dict, "data/data_dict4.pth")

    # s = "START 0 TOTAL:1.00, START TOTAL: 1.00 END"
    # rs = regex.search(r"(\sTOTAL.*)(1.00)(\s)", s)
    # for i in range(len(rs)):
    #     print(repr(rs[i]), rs.span(i))
