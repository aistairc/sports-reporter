import os
import time
import datetime
import json
import pickle
import click
import nltk
from tqdm import tqdm

import dynet_config
dynet_config.set(autobatch=1, mem=7544)
dynet_config.set_gpu()

from vocab import WordVocab, TableVocab
from trainer import Trainer
from network import Reporter
from utils import make_table, vectorize


@click.group()
def cli():
    pass


@cli.command()
@click.argument("vocab_file", type=click.Path(exists=True))
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--no_prog", is_flag=True)
def decode(vocab_file, model_file, input_file, no_prog):
    d = pickle.load(open(vocab_file, "rb"))
    wv = WordVocab.from_dump(d["vocab"]["word"])
    tv = {k: TableVocab.from_dump(v) for k, v in d["vocab"]["table"].items()}
    writer = d["author"] if "writer" in model_file else None
    model = Reporter.parse_config(tv=tv, wv=wv, writer=writer, model_file=model_file)

    inputs = json.load(open(input_file))
    for ins in tqdm(inputs, total=len(inputs), ncols=80, disable=no_prog):
        print(model.decode(make_table(ins), writer=writer.get(ins.get("author"), 0) if writer else None))


@cli.command()
@click.argument("vocab_file", type=click.Path(exists=True))
@click.option("--valid_file", type=click.Path(exists=True))
@click.option("--nh_vocab", type=click.INT, default=128)
@click.option("--nh_rnn", type=click.INT, default=512)
@click.option("--writer", is_flag=True)
@click.option("--learning_rate", "-lr", default=2e-3)
@click.option("--lr_decay", default=0.99)
@click.option("--batch_size", "-bs", default=16)
@click.option("--n_epoch", default=30)
@click.option("--log_dir", default="/tmp")
def train(vocab_file, valid_file, nh_vocab, nh_rnn, writer, learning_rate, lr_decay, batch_size, n_epoch, log_dir):
    log_dir = os.path.join(log_dir, str(int(time.time())))

    # Initialize...
    print(str(datetime.datetime.now()) + " Log dir at {}".format(log_dir))
    os.mkdir(log_dir)
    print(str(datetime.datetime.now()) + " Loading dataset...")
    d = pickle.load(open(vocab_file, "rb"))
    texts, tables = d["data"]["text"], d["data"]["table"]
    wv = WordVocab.from_dump(d["vocab"]["word"])
    tv = {k: TableVocab.from_dump(v) for k, v in d["vocab"]["table"].items()}
    writer = d["author"] if writer else None

    print(str(datetime.datetime.now()) + " Vectorizing...")
    data = list(vectorize(texts, tables, wv, tv, writer))

    valid = json.load(open(valid_file)) if valid_file else None

    # Model
    model = Reporter(tv=tv, wv=wv, nh_vocab=nh_vocab, nh_rnn=nh_rnn, writer=writer)
    print(str(datetime.datetime.now()) + " Model configurations...")
    print(str(datetime.datetime.now()) + " " + str(model))

    # Trainer
    trainer = Trainer(model, lr=learning_rate, decay=lr_decay, batch_size=batch_size)
    print(str(datetime.datetime.now()) + " Trainer configurations...")
    print(str(datetime.datetime.now()) + " " + str(trainer))

    try:
        best = 0.
        print(str(datetime.datetime.now()) + " Start training...")
        for _ in range(n_epoch):
            trainer.fit_partial(data)
            pc_name = str(model)+"_{}.dy".format(trainer.iter)
            model.pc.save(os.path.join(log_dir, pc_name))

            if valid and trainer.iter >= 5:
                pred = []
                prog = tqdm(desc="Evaluation: ", total=len(valid) + 1, ncols=80,)
                for ins in valid:
                    p = model.decode(make_table(ins), writer=writer.get(ins.get("author")) if writer else None)
                    pred.append(p.split())
                    prog.update()

                bleu = nltk.translate.bleu_score.corpus_bleu(
                    [[nltk.word_tokenize(' '.join(v["summary"]))] for v in valid], pred)
                prog.set_postfix(BLEU=bleu)
                prog.update()
                prog.close()
                if bleu > best:
                    best = bleu
                    print(str(datetime.datetime.now()) + " Save best model...")
                    model.pc.save(os.path.join(log_dir, str(model)+"_best.dy"))

    except KeyboardInterrupt:
        print("KeyboardInterrupted...")


if __name__ == '__main__':
    cli()
