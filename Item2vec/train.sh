#!/usr/bin/env bash
train_file = $1
item_vec_file=$2
../bin/word2vec -train $train_file -output $item_vec_file 128 -window 5 -sample le-3 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 100