import random
from difflib import SequenceMatcher
from string import ascii_uppercase, digits, punctuation

import numpy as np
import regex


def pred_to_dict(text, pred, prob):
    """Returns the prediction in a dictionary format."""
    resdict = {"company": ("", 0), "date": ("", 0), "address": ("", 0), "total": ("", 0)}
    keys = list(resdict)

    separators = [0] + (np.nonzero(numpy.diff(pred))[0] + 1).tolist() + [len(pred)]
    for i in range(len(separators) - 1):
        pred_class = pred[separators[i]] - 1
        if pred_class == -1:
            continue

        newkey = keys[pred_class]
        new_prob = prob[separators[i] : separators[i + 1]].max()
        if new_prob > res[newkey][1]:
            resdict[newkey] = (text[seps[i] : seps[i + 1]], new_prob)

    return {k: regex.sub(r"[\t\n]", " ", v[0].strip()) for k, v in resdict.items()}


def compare_truth(preddict, truthdict):
    """ Takes two dictionaries, returns float """
    ratio = 0
    for k in truthdict:
        ratio += SequenceMatcher(None, truthdict[k], preddict[k]).ratio()

    return ratio / len(truthdict)


def robust_padding(texts, labels):
    """Takes a list of texts: texts, and
             a numpy.array(): labels
       Returns: Padded strings with pad value == maximum length of string in `texts`
                Padded numpy array.
       Right pad and left pad is generated randomly."""

    maxlen = max(len(t) for t in texts)

    for i, text in enumerate(texts):
        if len(text) == maxlen:
            continue

        padbefore = random.randint(0, maxlen - len(text))
        padafter = maxlen - padbefore - len(text)

        texts[i] = random_string(padbefore) + text + random_string(padafter)
        labels[i] = np.pad(
            labels[i], (padbefore, padafter), "constant", constant_values=0
        )


def random_string(n):
    """Given an integer (indicates level of abstraction), returns a random string"""
    if n == 0:
        return ""

    x = random.random()
    if x > 0.5:
        pad = " " * n
    elif x > 0.3:
        pad = "".join(random.choices(digits + " \t\n", k=n))
    elif x > 0.2:
        pad = "".join(random.choices(ascii_uppercase + " \t\n", k=n))
    elif x > 0.1:
        pad = "".join(random.choices(ascii_uppercase + digits + " \t\n", k=n))
    else:
        pad = "".join(
            random.choices(ascii_uppercase + digits + punctuation + " \t\n", k=n)
        )

    return pad


if __name__ == "__main__":
    pred = {"a": "qwertyuiop", "b": "asdfghjkl", "c": "zxcvbnm"}

    truth = {"a": "qwertyu iop", "b": "ascfghjkl ", "c": ""}

    print(compare_truth(pred, truth))
