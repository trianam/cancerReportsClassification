#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python
if [ $# -lt 2 ]
    then
        echo "Use: \"$0 corpus outputDir [vectorSize] [windowSize] [maxIter] [x_max]\""
        exit
fi
if [ $# -lt 3 ]
    then
        VECTOR_SIZE=300
    else
        VECTOR_SIZE=$3
fi
if [ $# -lt 4 ]
    then
        WINDOW_SIZE=15
    else
        WINDOW_SIZE=$4
fi
if [ $# -lt 5 ]
    then
        MAX_ITER=15
    else
        MAX_ITER=$5
fi
if [ $# -lt 6 ]
    then
        X_MAX=10
    else
        X_MAX=$6
fi

CORPUS=$1
VOCAB_FILE=$2/vocab.txt
COOCCURRENCE_FILE=tmp/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=tmp/cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=$2/vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
BINARY=2
NUM_THREADS=8

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       rm $SAVE_FILE.bin
       rm $COOCCURRENCE_FILE
       rm $COOCCURRENCE_SHUF_FILE
    fi
  fi
fi

