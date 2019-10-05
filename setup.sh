#!/usr/bin/env bash

# Setup Modified RotoWire

git clone https://github.com/harvardnlp/boxscore-data.git
tar -xzvf boxscore-data/rotowire.tar.bz2
git clone https://github.com/aistairc/rotowire-modified

cd rotowire-modified
python script/generate_rotowire_modified.py --src_dir ../rotowire
cd ..
mv rotowire-modified/rotowire-modified rotowire_v2

rm -rf data2text-1 boxscore-data rotowire

# Setup IE model

git clone https://github.com/ratishsp/data2text-1.git
cd data2text-1
wget https://plu-aist.s3-ap-northeast-1.amazonaws.com/sports-reporter/ie-rotowire-modified.tar.gz
tar -xzvf ie-rotowire-modified.tar.gz
patch -p1 < rotowire-modified-ie.patch

rm ie-rotowire-modified.tar.gz
cd ..


# Download vocab file and model dump
wget https://plu-aist.s3-ap-northeast-1.amazonaws.com/sports-reporter/dump.tar.gz
tar -xzvf dump.tar.gz
rm dump.tar.gz

# To decode
# > python reporter.py decode data.pkl Reporter_nh_vocab-128_nh_rnn-512_26.dy
# > python reporter.py decode data.pkl Reporter_nh_vocab-128_nh_rnn-512_writer_15.dy