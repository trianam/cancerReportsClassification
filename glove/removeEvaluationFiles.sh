#!/bin/bash

#declare -a corpuses=("aladarV5a" "aladarV5b" "wikiIt")
declare -a corpuses=("wikiIt")
declare -a vectorSizes=(200 300 400)
declare -a windowSizes=(10 15 20)
declare -a iterations=(15 25 40)
declare -a topNumbers=(1 5 10 15)
xMax=100

OUTPATH=auto

for c in ${corpuses[@]}
do
    for v in ${vectorSizes[@]}
    do
        for w in ${windowSizes[@]}
        do
            for i in ${iterations[@]}
            do
                currPath=$OUTPATH/iter$i-xMax$xMax/out-$c-$v-$w
                for t in ${topNumbers[@]}
                do
                    outputFile=$currPath/evalTOP$t.txt
                    if [ -f $outputFile ]; then
                        rm $outputFile
                        echo "Removed $outputFile"
                    fi
                done
            done
        done
    done
done

