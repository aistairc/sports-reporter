import re
from collections import defaultdict
import numpy as np
import dynet as dy

import nn
from utils import NUM_ATTR, NUM2TEXT


class Encoder:
    def __init__(self, vocab, nh_table, model):
        self.pc = pc = model
        self.v = vocab
        self.nh_table = nh_table

        self.r = pc.add_lookup_parameters((len(vocab.r2i), nh_table))  # Subject
        self.a = pc.add_lookup_parameters((len(vocab.a2i), nh_table))  # Attribute
        self.c = {i: pc.add_lookup_parameters((len(vocab.c2i[a]), nh_table)) for a, i in vocab.a2i.items()}  # Object
        self.b = pc.add_lookup_parameters((2, nh_table))  # IS_HOME

        # Cell encoding
        self.ce = nn.Sequential(nn.Linear(4 * nh_table, nh_table, pc), dy.tanh)  # Subject, Attribute, Object, IS_HOME
        # Row encoding
        self.re = nn.Sequential(nn.Linear(vocab.n_attr * nh_table, nh_table, pc), dy.tanh)

    def initialize(self, table, r2i):
        cells = defaultdict(list)
        for a in range(len(self.v.attr)):
            for r, (c, is_home) in table[a].items():
                ce = self.ce(dy.concatenate([self.r[r2i[r]], self.a[a], self.c[a][c], self.b[is_home]]))
                cells[r].append(ce)

        keys = tuple(cells)
        rows = {k: self.re(dy.concatenate(cells[k])) for k in keys}  # Hidden, Row
        cells = dy.concatenate_cols([c for k in keys for c in cells[k]])  # Hidden, Col * Row

        return keys, cells, rows


class Reporter:
    def __init__(self, tv, wv, nh_vocab=128, nh_rnn=512, writer=False):
        self.tv = tv
        self.wv = wv
        self.n_word = n_word = len(wv.w2i)
        self.nh_vocab = nh_vocab
        self.nh_rnn = nh_rnn

        pc = self.pc = dy.ParameterCollection()
        self.SOS = pc.add_parameters((2 * nh_vocab,))
        self.EOS = wv.EOS
        self.writer = pc.add_lookup_parameters((len(writer), nh_rnn)) if isinstance(writer, dict) else None

        self.Encoders = {k: Encoder(v, nh_vocab, pc) for k, v in tv.items()}

        self.Ref = nn.Linear(2 * nh_rnn, 1, pc)
        self.New = nn.BiLinear(nh_vocab, nh_rnn, pc)
        self.Obs = nn.BiLinear(nh_rnn, nh_rnn, pc)
        self.Pointer = nn.BiLinear(nh_vocab, 2 * nh_rnn, pc)

        self.Row = dy.GRUBuilder(1, nh_vocab, nh_rnn, pc)
        self.Col = dy.GRUBuilder(1, nh_vocab, nh_rnn, pc)
        self.Re = nn.Sequential(nn.Linear(nh_rnn, nh_vocab, pc), dy.tanh)

        self.w = pc.add_lookup_parameters((n_word, nh_vocab))
        self.LM = dy.LSTMBuilder(2, 2 * nh_vocab, nh_rnn, pc)
        self.H = nn.Sequential(nn.Linear((3 if writer else 2) * nh_rnn, nh_vocab, pc), dy.tanh)
        self.Out = dy.StandardSoftmaxBuilder(nh_vocab, len(wv.w2i), pc)

        self.NUM = nn.Linear(2 * nh_rnn, 1, pc)
        self.REFRESH = pc.add_parameters((nh_vocab,))

    def __repr__(self):
        s = self.__class__.__name__ + "_nh_vocab-{}_nh_rnn-{}".format(self.nh_vocab, self.nh_rnn)
        if self.writer:
            s += "_writer"
        return s

    @classmethod
    def parse_config(cls, tv, wv, writer, model_file):
        def ptn(key):
            return int(re.search(key + "-([0-9]+)", model_file).group(1))

        model = Reporter(tv, wv, nh_vocab=ptn("nh_vocab"), nh_rnn=ptn("nh_rnn"), writer=writer)
        model.pc.populate(model_file)
        return model

    def pick_cell(self, cells, keys, idx):
        total = 0
        for i, k in enumerate(keys):
            n_attr = self.tv["player" if k.isdigit() else "team"].n_attr

            if k == idx:
                return dy.pick_range(cells, total, total + n_attr, d=1)
            total += n_attr

    def initialize(self, tables):
        cells, idx = [], []
        keys = tuple()
        rows, tracker = {}, {}
        x_0 = []
        for tbl_n in self.Encoders:
            k, cs, rs = self.Encoders[tbl_n].initialize(*(t[tbl_n] for t in tables))
            keys += k
            cells.append(cs), rows.update(rs)
            idx.extend([(tbl_n, k) for k in cs])
            x_0.append(dy.mean_dim(cs, d=[1], b=False))
        x_0 = dy.mean_dim(dy.concatenate_cols(x_0), d=[1], b=False)

        tracker["EMPTY"] = self.Row.initial_state().add_input(x_0)

        lm = self.LM.initial_state().add_input(self.SOS)
        return keys, dy.concatenate_cols(cells), rows, tracker, lm

    def loss(self, tables, texts, writer=None):
        keys, cells, rows, tracker, lm = self.initialize(tables)

        latest = "EMPTY"
        for sent in texts + [[(self.EOS, None)]]:
            for i, (word, entity) in enumerate(sent, 1):
                ref_score = self.Ref(dy.concatenate([lm.output(), tracker[latest].output()]))  # Current Entity or not
                yield - dy.log_sigmoid(ref_score) + (0 if entity else ref_score)

                if entity:
                    tbl_n, attr, idx = entity
                    new_key = tuple(rows)
                    new_score = self.New(dy.concatenate_cols([rows[k] for k in new_key]), lm.output()) if new_key else None

                    if latest != "EMPTY":
                        obs_key = tuple(tracker)
                        obs_score = self.Obs(dy.concatenate_cols([tracker[k].output() for k in obs_key]), lm.output())
                        ent_key = new_key + obs_key
                        ent_score = dy.concatenate([new_score, obs_score]) if new_score else obs_score
                    else:
                        ent_key, ent_score = new_key, new_score

                    yield dy.pickneglogsoftmax(ent_score, ent_key.index(idx))

                    if idx not in tracker:
                        tracker[idx] = self.Row.initial_state([tracker[latest].output()]).add_input(rows.pop(idx))
                        if "EMPTY" in tracker:
                            del tracker["EMPTY"]
                    elif idx != latest:
                        tracker[idx] = self.Row.initial_state([tracker[latest].output()]).add_input(self.Re(tracker[idx].output()))
                    latest = str(idx)

                    ent_cell = self.pick_cell(cells, keys, idx)
                    pick_score = self.Pointer(ent_cell, dy.concatenate([lm.output(), tracker[latest].output()]))
                    yield dy.pickneglogsoftmax(pick_score, attr)
                    tracker[idx] = self.Col.initial_state([tracker[idx].output()]).add_input(dy.pick(ent_cell, attr, dim=1))

                    if self.tv[tbl_n].i2a[attr] in NUM_ATTR:
                        num_score = self.NUM(dy.concatenate([lm.output(), tracker[latest].output()]))
                        yield - dy.log_sigmoid(num_score) + (0 if self.wv.i2w[word].isdigit() else num_score)

                hs = [lm.output(), tracker[latest].output()] + ([self.writer[writer]] if self.writer is not None else [])
                h = self.H(dy.concatenate(hs))
                lm = lm.add_input(dy.concatenate([h, self.w[word]]))

                if not entity:
                    yield self.Out.neg_log_softmax(h, word)

                if i == len(sent):
                    tracker[latest] = tracker[latest].add_input(self.REFRESH)

    def decode(self, tables, writer=None):
        dy.renew_cg()
        tbl_vec, r2i = {}, {}
        for tbl_n, vocab in self.tv.items():
            tbl_vec[tbl_n], r2i[tbl_n] = vocab.vectorize(tables[tbl_n])

        keys, cells, rows, tracker, lm = self.initialize((tbl_vec, r2i))

        doc, latest = [], "EMPTY"
        while True:
            ref_score = self.Ref(dy.concatenate([lm.output(), tracker[latest].output()]))
            is_ref = bool(ref_score.value() > 0.)

            if is_ref:
                new_key = tuple(rows)
                new_score = self.New(dy.concatenate_cols([rows[k] for k in new_key]), lm.output()) if new_key else None
                if latest != "EMPTY":
                    obs_key = tuple(tracker)
                    obs_score = self.Obs(dy.concatenate_cols([tracker[k].output() for k in obs_key]), lm.output())
                    ent_key = new_key + obs_key
                    ent_score = dy.concatenate([new_score, obs_score]) if new_score else obs_score
                else:
                    ent_key, ent_score = new_key, new_score

                idx = ent_key[int(np.argmax(ent_score.npvalue()))]
                tbl_n = "player" if idx.isdigit() else "team"

                if idx not in tracker:
                    tracker[idx] = self.Row.initial_state([tracker[latest].output()]).add_input(rows.pop(idx))
                    if "EMPTY" in tracker:
                        del tracker["EMPTY"]
                elif idx != latest:
                    tracker[idx] = self.Row.initial_state([tracker[latest].output()]).add_input(
                        self.Re(tracker[idx].output()))
                latest = str(idx)

                ent_cell = self.pick_cell(cells, keys, idx)
                pick_score = self.Pointer(ent_cell, dy.concatenate([lm.output(), tracker[latest].output()]))
                attr_idx = np.argmax(pick_score.npvalue())
                tracker[idx] = self.Col.initial_state([tracker[idx].output()]).add_input(
                    dy.pick(ent_cell, attr_idx, dim=1))

                attr = self.tv[tbl_n].i2a[attr_idx]

                if attr in NUM_ATTR:
                    num_score = self.NUM(dy.concatenate([lm.output(), tracker[latest].output()]))
                    is_digit = bool(num_score.value() > 0.)
                    num = tables[tbl_n][attr][idx]
                    word = num if is_digit else NUM2TEXT.get(num, num)

                else:
                    word = tables[tbl_n][attr][idx]

                hs = [lm.output(), tracker[latest].output()] + ([self.writer[writer]] if self.writer is not None else [])
                h = self.H(dy.concatenate(hs))

            else:
                hs = [lm.output(), tracker[latest].output()] + ([self.writer[writer]] if self.writer is not None else [])
                h = self.H(dy.concatenate(hs))

                logits = self.Out.full_logits(h).npvalue()
                if len(doc) < 150:
                    logits[self.EOS] = - np.inf
                word_id = np.argmax(logits)

                if word_id == self.EOS or len(doc) >= 1500:
                    break
                word = self.wv.i2w[word_id]

            doc.append(word)
            if doc[-1] == ".":
                tracker[latest] = tracker[latest].add_input(self.REFRESH)

            lm = lm.add_input(dy.concatenate([h, self.w[self.wv.w2i.get(word, self.wv.UNK)]]))

        return " ".join(doc)
