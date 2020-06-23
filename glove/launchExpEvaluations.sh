#!/bin/bash

declare -a corpuses=("aladarV7")
#declare -a vectorSizes=(100 200 300 400)
#declare -a windowSizes=(8 10 15 20 25)
#declare -a iterations=(20 40 50)
#declare -a xMaxs=(5 10 50 100)

#declare -a vectorSizes=(10 25 50)
#declare -a windowSizes=(2 4 6 8 10)
#declare -a iterations=(10 20 40)
#declare -a xMaxs=(2 5 10)

#declare -a vectorSizes=(60 70 80 90 100)
#declare -a windowSizes=(6 8 10)
#declare -a iterations=(10 20 40 50)
#declare -a xMaxs=(10 100)

declare -a vectorSizes=(30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 300)
declare -a windowSizes=(15)
declare -a iterations=(50)
declare -a xMaxs=(100)

declare -a topNumbers=(5 10 15)

OUTPATH=auto

n=0
tt=$[ ${#corpuses[@]}*${#vectorSizes[@]}*${#windowSizes[@]}*${#iterations[@]}*${#xMaxs[@]} ]

for c in ${corpuses[@]}
do
    for v in ${vectorSizes[@]}
    do
        for w in ${windowSizes[@]}
        do
            for i in ${iterations[@]}
            do
		for x in ${xMaxs[@]}
		do
		    n=$[ $n+1 ]
                    echo "======================================"
                    echo -e "     \tcorpus   \tvecs\twins\titer\txMax"
                    echo -e "START\t$c\t$v\t$w\t$i\t$x\t($n/$tt)"
                    echo "--------------------------------------"

		    currPath=$OUTPATH/iter$i-xMax$x/out-$c-$v-$w
		    vocabFile=$currPath/vocab.txt
		    vectorFile=$currPath/vectors.txt
		    outputFile=$currPath/evalTOP1.txt

		    if [ ! -f $vocabFile ]; then
			echo "Vocabulary file $vocabFile do not exists!"
			exit 1
		    fi
		    if [ ! -f $vectorFile ]; then
			echo "Vectors file $vectorFile do not exists!"
			exit 1
		    fi
		    if [ -f $outputFile ]; then
			echo "Output file $outputFile already exists!"
		    else
			python evalAla/evaluateTOP1.py --vocab_file $vocabFile --vectors_file $vectorFile --output_file $outputFile
			for t in ${topNumbers[@]}
			do
			echo "---------------------------"
			    outputFile=$currPath/evalTOP$t.txt
			    if [ -f $outputFile ]; then
				echo "Output file $outputFile already exists!"
				exit 2
			    fi
			    python evalAla/evaluate.py --vocab_file $vocabFile --vectors_file $vectorFile --top_num $t --output_file $outputFile
			done
		    fi
		done
            done
        done
    done
done

