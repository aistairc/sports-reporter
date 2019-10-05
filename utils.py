import nltk
from collections import defaultdict
from text2num import text2num

NUM_ATTR = {'TEAM-PTS_QTR2', 'TEAM-FT_PCT', 'TEAM-PTS_QTR1', 'TEAM-PTS_QTR4', 'TEAM-PTS_QTR3', 'TEAM-PTS', 'TEAM-AST',
            'TEAM-LOSSES', 'TEAM-WINS', 'TEAM-REB', 'TEAM-TOV', 'TEAM-FG3_PCT', 'TEAM-FG_PCT', 'MIN', 'FGM', 'REB',
            'FG3A', 'AST', 'FG3M', 'OREB', 'TO', 'PF', 'PTS', 'FGA', 'STL', 'FTA', 'BLK', 'DREB', 'FTM', 'FT_PCT',
            'FG_PCT', 'FG3_PCT'}

NUM2TEXT = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
            '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen',
            '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
            "N/A": "0"}


def vectorize(texts, tables, wv, tv, writer):
    for doc, tbl in zip(texts, tables):
        entities = {r[0] for sent, rel in doc for r in rel}
        if entities:
            table_vectors, t2r2i = {}, {}
            for tbl_n, v in tv.items():
                table, r2i = v.vectorize(tbl[tbl_n])
                table_vectors[tbl_n] = table
                t2r2i[tbl_n] = r2i

            doc_ids, sent_ids = [], []
            for sent, rel in doc:
                sent_ids = [[wv.w2i.get(w, wv.UNK), None] for w in sent]
                for idx, attr, loc in rel:
                    tbl_n = "team" if attr.startswith("TEAM") else "player"
                    sent_ids[loc][1] = (tbl_n, tv[tbl_n].a2i[attr], idx)

                assert len(sent) == len(sent_ids)
                doc_ids.append(sent_ids)
            if doc_ids:
                yield (table_vectors, t2r2i), doc_ids, writer.get(tbl.get("author"), 0) if writer else None


def get_ents(ins):
    teams, cities, players = set(), set(), set()
    for k in ("home", "vis"):
        city = ins[k + "_city"]
        teams.update({
            ins[k + "_name"],
            city + " " + ins[k + "_name"],
            ins[k + "_line"]["TEAM-NAME"],
            city + " " + ins[k + "_line"]["TEAM-NAME"]
        })
        cities.add(city)
        if city == "Los Angeles":
            cities.add("LA")
        elif city == "LA":
            cities.add("Los Angeles")

    players.update(ins["box_score"]["PLAYER_NAME"].values())
    players.update([" ".join(v.split(" ")[:-1]) for v in ins["box_score"]["PLAYER_NAME"].values()])

    for ents in [teams, cities, players]:
        for k in list(ents):
            pieces = k.split() + k.split("-")
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        ents.add(piece)
    return teams, cities, players


def table_concat(tables):
    t = defaultdict(list)
    for ins in tables:
        for k, v in ins.items():
            if k != "author":
                t[k].append(v)
    return t


def make_table(ins):
    home_city = ins["home_line"]["TEAM-CITY"]
    team = {}
    for k in ins["home_line"].keys():
        team[k] = {"home": ins["home_line"][k], "vis": ins["vis_line"][k]}
    team["IS_HOME"] = {"home": 1, "vis": 0}

    bs = ins["box_score"]
    bs["IS_HOME"] = {idx: int(value == home_city) for idx, value in bs["TEAM_CITY"].items()}
    return {"team": team, "player": bs, "author": ins.get("author")}


def extract_entities(sent, teams, cities, players, ins, rel):
    entities = []
    i = 0
    while i < len(sent):
        if sent[i] in teams | cities | players:
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in teams | cities | players:
                j += 1
            for loc in range(i, i + j - 1):
                if sent[loc] in cities:
                    if sent[loc] == "LA":
                        is_home = ins["home_line"]["TEAM-CITY"] in ("LA", "Los Angeles")
                        entities.append(("home" if is_home else "vis", "TEAM-CITY", loc))
                    else:
                        if sent[loc] in ins["home_line"]["TEAM-CITY"].split(" "):
                            entities.append(("home", "TEAM-CITY", loc))
                        elif sent[loc] in ins["vis_line"]["TEAM-CITY"].split(" "):
                            entities.append(("vis", "TEAM-CITY", loc))

                elif sent[loc] in teams:
                    if sent[loc] in ins["home_line"]["TEAM-NAME"].split(" "):
                        entities.append(("home", "TEAM-NAME", loc))
                    elif sent[loc] in ins["vis_line"]["TEAM-NAME"].split(" "):
                        entities.append(("vis", "TEAM-NAME", loc))

                elif sent[loc] in players:
                    if sent[loc] in ins["box_score"]["SECOND_NAME"].values():
                        list_of_player = {k for k, v in ins["box_score"]["SECOND_NAME"].items() if v == sent[loc]}
                        if len(list_of_player) > 1:
                            inter = {r[0] for r in rel} & list_of_player
                            if inter:
                                entities.append((inter.pop(), "SECOND_NAME", loc))
                        else:
                            entities.append((list_of_player.pop(), "SECOND_NAME", loc))
                    elif sent[loc] in ins["box_score"]["FIRST_NAME"].values():
                        list_of_player = {k for k, v in ins["box_score"]["FIRST_NAME"].items() if v == sent[loc]}
                        if len(list_of_player) > 1:
                            inter = {r[0] for r in rel} & list_of_player
                            if inter:
                                entities.append((inter.pop(), "FIRST_NAME", loc))
                        entities.append((list_of_player.pop(), "FIRST_NAME", loc))
            i += j - 1
        else:
            i += 1
    return entities


def make_text(data, tuples):
    i, j = -1, 0
    relations, rel, r = [], [], []
    teams, cities, players, bs = set(), set(), set(), dict()
    doc = []
    for line in open(tuples):
        line = line.rstrip()
        if line:
            entity, value, attribute, loc = line.split("|")
            if value == "UNK":
                continue
            loc = int(loc)
            entity = entity.replace("UNK", "").strip()
            es = set(entity.split(" ") + entity.split("-"))
            if any(e in teams | cities for e in es):
                home = set(data[i]["home_name"].split(" ") + data[i]["home_city"].split(" "))
                if "LA" in home:
                    home.update({"Los", "Angeles"})
                elif "Los" in home:
                    home.add("LA")
                vis = set(data[i]["vis_name"].split(" ") + data[i]["vis_city"].split(" "))
                if "LA" in vis:
                    home.update({"Los", "Angeles"})
                elif "Los" in vis:
                    home.add("LA")

                n_home = sum(1 for e in entity.split(" ") if e in home)
                n_vis = sum(1 for e in entity.split(" ")if e in vis)
                assert n_home or n_vis
                ix = "home" if n_home > n_vis else "vis"
            elif attribute.startswith("PLAYER-") and any(e in players for e in es):
                ix = set()
                for key in ("SECOND_NAME", "PLAYER_NAME", "FIRST_NAME"):
                    for k, v in bs[key].items():
                        if entity == v:
                            ix.add(k)
                        for vs in v.split(" "):
                            if entity == vs:
                                ix.add(k)
                if len(ix) == 1:
                    ix = ix.pop()
                elif len(ix) > 1:
                    num_text = value if value.isdigit() else str(text2num(value))

                    ixx = {i for i in ix if bs[attribute[7:]][i] == num_text}
                    if not ixx:
                        ixx = {i for i in ix if bs[attribute[7:]][i] != "N/A" and int(bs[attribute[7:]][i]) > 0}
                    if len(ixx) == 1:
                        ix = ixx.pop()
                    elif len(ixx) > 1:
                        ix &= set(i for i, _, _ in r)
                        if len(ix) == 1:
                            ix = ixx.pop()
                        else:
                            continue
                    else:
                        continue
                else:
                    if entity:
                        if entity == "Jefferson":
                            entity = "Hollis-" + entity
                        elif entity[:4] == "Will":
                            entity = "Carter-" + entity

                        ents = entity.split(" ")
                        for key in ("SECOND_NAME", "PLAYER_NAME", "FIRST_NAME"):
                            for k, v in bs[key].items():
                                if v in ents:
                                    ix.add(k)
                                for vs in v.split(" "):
                                    if vs in ents:
                                        ix.add(k)
                        if len(ix) == 1:
                            ix = ix.pop()
                        else:
                            continue
                    else:
                        continue
            else:
                if attribute.startswith("PLAYER"):
                    ix = {k for k, v in bs[attribute[7:]].items() if v == value}
                    if len(ix) == 1:
                        ix = ix.pop()
                    else:
                        continue
                else:
                    continue

            while loc >= len(doc[j]) or doc[j][loc] != value:
                ents = extract_entities(doc[j], teams, cities, players, data[i], r)
                rs = [list(r) for r in sorted(set(r + ents), key=lambda x: x[-1])]
                rel.append((doc[j], rs))

                j += 1
                r = []
                if len(doc) == j:
                    relations.append(rel)
                    i += 1
                    j = 0
                    if i == len(data):
                        return relations[1:]
                    raw = ' '.join(nltk.word_tokenize(' '.join(data[i]['summary'])))
                    doc = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(raw)]
                    rel = []
            assert doc[j][loc] == value
            r.append((ix, attribute.replace("PLAYER-", ""), loc))
        else:
            while j < len(doc):
                ents = extract_entities(doc[j], teams, cities, players, data[i], r)
                rs = [list(r) for r in sorted(set(r + ents), key=lambda x: x[-1])]
                rel.append((doc[j], rs))
                r = []
                j += 1
            i += 1
            j = 0
            assert len(rel) == len(doc)

            for sent, rss in rel:
                m = 0
                while m + 1 < len(rss):
                    if rss[m + 1][-1] - rss[m][-1] == 1 and rss[m][:-1] == rss[m + 1][:-1]:
                        sent[rss[m][-1]] += "_" + sent.pop(rss[m + 1][-1])
                        for rr in rss[m + 1:]:
                            rr[-1] -= 1
                        del rss[m + 1]
                    m += 1

            relations.append(rel)
            rel, r = [], []
            teams, cities, players = get_ents(data[i])
            bs = data[i]["box_score"]

            raw = ' '.join(nltk.word_tokenize(' '.join(data[i]['summary'])))
            doc = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(raw)]

    relations.append(rel)
    return relations[1:]
