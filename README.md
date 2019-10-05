# Sports Reporter

Python code for Learning to Select, Track, and Generate for Data-to-Text (Iso et al; ACL 2019).


## Resources
### Rotowire-modified dataset
Please refer to [rotowire-modified](https://github.com/aistairc/rotowire-modified) repo.

## Usage

## Preprocessing
```
./setup.sh
DATA=<path to the rotowire-modified directory>
ANNOTAION=<path to the text annotation file for training>
VOCAB=<path to the vocablary file>
python make_data.py $DATA $ANNOTATION $VOCAB
```
The annotation file could be obtained from information extractor.

### Train model
```
python reporter.py train $VOCAB --valid_file $DATA/valid.json
```

### Decode
```
MODEL=<path to the trained model file>
python reporter.py decode $VOCAB $MODEL $DATA/test.json
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
This code is available under the MIT Licence, see LICENCE

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
```