import random
from difflib import SequenceMatcher
from string import ascii_uppercase, digits, punctuation

import numpy as np
import regex


def pred_to_dict(text, pred, prob):
    """Returns the prediction in a dictionary format."""
    res_dict = {"company": ("", 0), "date": ("", 0), "address": ("", 0), "total": ("", 0)}
    keys = list(res_dict)

    seprtrs = [0] + (np.nonzero(numpy.diff(pred))[0] + 1).tolist() + [len(pred)]
    for i in range(len(seprtrs) - 1):
        pred_class = pred[seprtrs[i]] - 1
        if pred_class == -1:
            continue

        new_key = keys[pred_class]
        new_prob = prob[seprtrs[i] : seprtrs[i + 1]].max()
        if new_prob > res[new_key][1]:
            res_dict[new_key] = (text[seps[i] : seps[i + 1]], new_prob)

    return {k: regex.sub(r"[\t\n]", " ", v[0].strip()) for k, v in res_dict.items()}


def compare_truth(pred_dict, truth_dict):
    """ Takes two dictionaries, returns float """
    ratio = 0
    for k in truth_dict:
        ratio += SequenceMatcher(None, truth_dict[k], pred_dict[k]).ratio()

    return ratio / len(truth_dict)


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

        pad_before = random.randint(0, maxlen - len(text))
        pad_after = maxlen - pad_before - len(text)

        texts[i] = random_string(pad_before) + text + random_string(pad_after)
        labels[i] = np.pad(
            labels[i], (pad_before, pad_after), "constant", constant_values=0
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
