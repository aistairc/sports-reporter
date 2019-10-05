from collections import defaultdict


class WordVocab:
    def __init__(self, ws):
        w2i = defaultdict(lambda: len(w2i))
        self.UNK = w2i["<UNK>"]
        self.SOS = w2i["<SOS>"]
        self.EOS = w2i["<EOS>"]
        for w in ws:
            _ = w2i[w]
        self.w2i = dict(w2i)
        self.i2w = {v: k for k, v in self.w2i.items()}

    @classmethod
    def from_dump(cls, dump):
        vocab = cls.__new__(cls)
        vocab.__dict__.update(dump)
        return vocab

    def __getitem__(self, item):
        return self.w2i.get(str(item).replace(" ", "_"), self.UNK)

    def vecotorize(self, texts):
        return [self.w2i[w] for w in texts]


class TableVocab:
    def __init__(self, tables, key="player", min_count=1):
        self.key = key
        rows, attr, cells = defaultdict(int), set(), defaultdict(lambda: defaultdict(int))

        for table in tables:
            for v in table["PLAYER_NAME" if key == "player" else "TEAM-NAME"].values():
                rows[v] += 1
        r2i = {k: i for i, k in enumerate({k for k, v in rows.items() if v > min_count}, 1)}
        r2i["<UNK>"] = self.UNK = 0
        self.r2i = dict(r2i)

        for table in tables:
            for a, vs in table.items():
                if a not in ("PLAYER_NAME", "IS_HOME"):
                    attr.add(a)
                    for v in vs.values():
                        cells[a][v] += 1

        self.attr = attr
        self.a2i = {k: i for i, k in enumerate(attr)}
        self.i2a = {v: k for k, v in self.a2i.items()}
        self.n_attr = len(attr)

        self.c2i = {}
        for a, vs in cells.items():
            self.c2i[a] = {k: i for i, k in enumerate({k for k, v in vs.items() if v > min_count}, 1)}
            self.c2i[a]["<UNK>"] = self.UNK

    @classmethod
    def from_dump(cls, dump):
        vocab = cls.__new__(cls)
        vocab.__dict__.update(dump)
        return vocab

    def vectorize(self, table):
        rows = {k: self.r2i.get(v, self.UNK) for k, v in
                table["PLAYER_NAME" if self.key == "player" else "TEAM-NAME"].items()}
        vec = dict()
        for a, vs in table.items():
            if a not in ("PLAYER_NAME", "IS_HOME"):
                vec[self.a2i[a]] = {idx: (self.c2i[a].get(v, self.UNK), table["IS_HOME"][idx]) for idx, v in vs.items()}
        return vec, rows
