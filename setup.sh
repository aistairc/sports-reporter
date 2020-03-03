#!/usr/bin/env bash

# Setup Modified RotoWire

git clone https://github.com/harvardnlp/boxscore-data.git
tar -jxvf boxscore-data/rotowire.tar.bz2
git clone https://github.com/aistairc/rotowire-modified

cd rotowire-modified
python script/generate_rotowire_modified.py --src_dir ../rotowire
cd ..
mv rotowire-modified/rotowire-modified rotowire_v2

rm -rf boxscore-data rotowire

# Setup IE model
git clone https://github.com/ratishsp/data2text-1.git
cd data2text-1
cat ../ie/ie-rotowire-modified.z0* > ie-rotowire-modified.tar.gz
tar -xzvf ie-rotowire-modified.tar.gz --strip=1
patch -p1 < rotowire-modified-ie.patch

rm ie-rotowire-modified.tar.gz
cd ..

# Download vocab file and model dump
cat ./dump/dump.z0* > dump.tar.gz
tar -xzvf dump.tar.gz
rm dump.tar.gz

# To decode
# > python reporter.py decode data.pkl Reporter_nh_vocab-128_nh_rnn-512_26.dy
# > python reporter.py decode data.pkl Reporter_nh_vocab-128_nh_rnn-512_writer_15.dy
