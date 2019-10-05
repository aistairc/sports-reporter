import os
import json
import click
import datetime
import pickle

from vocab import TableVocab, WordVocab
from utils import make_table, make_text


@click.command()
@click.argument("dir_path", type=click.Path(exists=True))
@click.argument("plan_path", type=click.Path(exists=True))
@click.argument("out_path")
def prep(dir_path, plan_path, out_path):
    desc = str(datetime.datetime.now()) + " Overwrite the preprocessed data: {}? Y/n (default: n)".format(out_path)
    if os.path.exists(out_path) and input(desc) != "Y":
        print(str(datetime.datetime.now()) + " Exit.")
        exit()

    print(str(datetime.datetime.now()) + " Building dataset from " + dir_path)

    train = json.load(open(os.path.join(dir_path, "train.json")))
    tables, texts = [make_table(ins) for ins in train], list(make_text(train, plan_path))
    authors = {ins.get("author", "UNK") for ins in train}
    assert len(tables) == len(texts)

    tv = {}
    for k in ("team", "player"):
        tv[k] = TableVocab([t[k] for t in tables], key=k)
    wv = WordVocab({w for doc in texts for sent, _ in doc for w in sent})

    print(str(datetime.datetime.now()) + " Saving dataset from " + out_path)
    pickle.dump({
        "data": {"text": texts, "table": tables},
        "vocab": {"word": wv.__dict__, "table": {k: v.__dict__ for k, v in tv.items()}},
        "author": {k: i for i, k in enumerate(authors)}
    }, open(out_path, "wb"))


if __name__ == '__main__':
    prep()
