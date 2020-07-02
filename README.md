# Sports Reporter

[![Conference](https://img.shields.io/badge/acl-2019-red)](https://www.aclweb.org/anthology/P19-1202/)
[![arXiv](https://img.shields.io/badge/arxiv-1907.09699-success)](https://arxiv.org/abs/1907.09699)
[![Poster](https://img.shields.io/badge/poster-pdf-informational)](https://isomap.github.io/posters/iso2019acl.pdf)

Python code for Learning to Select, Track, and Generate for Data-to-Text (Iso et al; ACL 2019).


## Resources
### Rotowire-modified dataset
Please refer to [rotowire-modified](https://github.com/aistairc/rotowire-modified) repo.

## Usage

## Dependencies
- The code was written for Python 3.X and requires DyNet.
- Dependencies can be installed using `requirements.txt`.
- For running information extractor, you should install [torch](http://torch.ch/).

## Preprocessing
Before starting an experiment, you should run our provided `setup.sh`.
```
./setup.sh
```

After that, you can make the annotation file for training data via information extractor:
```
cd ./data2text-1
cat ../rotowire_v2/train.json | python -c 'import sys, json, nltk; print("\n".join(" ".join(nltk.word_tokenize(" ".join(x["summary"]))) for x in json.load(sys.stdin)))' > ../rotowire_v2/train_summary.txt
python data_utils.py -mode prep_gen_data -gen_fi ../rotowire_v2/train_summary.txt -dict_pfx "rotowire-modified-ie" -output_fi train_gold.h5 -input_path "../rotowire_v2" -train
th extractor.lua -gpuid 1 -datafile rotowire-modified-ie.h5 -preddata train_gold.h5 -dict_pfx "rotowire-modified-ie" -just_eval
```
Then, you can see the annotation file `train_gold.h5-tuples.txt` and make a vocab file for training.
```
cd ..
VOCAB=<path to the vocablary file>
python make_data.py ./rotowire_v2 ./data2text-1/train_gold.h5-tuples.txt $VOCAB
```

### Train model
```
python reporter.py train $VOCAB --valid_file ./rotowire_v2/valid.json
```

### Decode
```
MODEL=<path to the trained model file>
python reporter.py decode $VOCAB $MODEL ./rotowire_v2/test.json
```

## Updated Results for RotoWire-modified

| without writer info | RG (P% / #) | CS (P% / R%)| CO  |BLEU |
|---------------------|:-----------:|:-----------:|:---:|:---:|
|Joint+Rec+TVD (B=5)  |18.09 / 48.54|23.24 / 28/92|14.47|15.34|
|Conditional (B=5)    |20.28 / 61.76|27.20 / 29.76|15.88|15.26|
|Puduppully+, AAAI'19 |82.55 / 34.05|32.30 / 43.74|16.67|14.82|
|Puduppully+, ACL'19  |91.13 / 32.41|37.05 / 43.06|20.62|15.23|
|Iso+, ACL'19         |91.98 / 31.66|40.44 / 46.63|21.56|15.74|


| with writer info    | RG (P% / #) | CS (P% / R%)| CO  |BLEU |
|---------------------|:-----------:|:-----------:|:---:|:---:|
|Puduppully+, AAAI'19 |82.55 / 34.05|32.30 / 43.74|16.67|14.82|
|+ stage 1            |85.54 / 30.26|42.33 / 49.38|21.26|18.01|
|+ stage 2            |83.35 / 32.42|33.28 / 42.92|16.73|16.57|
|+ stage 1 & 2        |84.09 / 28.16|43.63 / 47.75|21.96|18.57|
|Iso+, ACL'19         |91.98 / 31.66|40.44 / 46.63|21.56|15.74|
|+ writer             |93.32 / 29.44|51.76 / 55.21|24.97|20.62|

## License and References
This code is available under the MIT Licence, see [LICENCE](https://github.com/aistairc/sports-reporter/blob/master/LICENCE)

When you write a paper using this code, please cite the followings.

```tex
@InProceedings{Iso2019Learning,
    author = {Iso, Hayate
              and Uehara, Yui
              and Ishigaki, Tatsuya
              and Noji, Hiroshi
              and Aramaki, Eiji
              and Kobayashi, Ichiro
              and Miyao, Yusuke
              and Okazaki, Naoaki
              and Takamura, Hiroya},
    title = {Learning to Select, Track, and Generate for Data-to-Text},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year = {2019}
  }
```


## Author
[@isomap](https://github.com/isomap/)
