#!/bin/bash

#declare -a corpuses=("aladarV7")
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

#declare -a vectorSizes=(30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 300)
#declare -a windowSizes=(15)
#declare -a iterations=(50)
#declare -a xMaxs=(100)

declare -a corpuses=("temporal")
declare -a vectorSizes=(60)
declare -a windowSizes=(15)
declare -a iterations=(50)
declare -a xMaxs=(100)


n=0
t=$[ ${#corpuses[@]}*${#vectorSizes[@]}*${#windowSizes[@]}*${#iterations[@]}*${#xMaxs[@]} ]

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
                    corpusFile=corpus/$c.txt
                    outPath=auto/iter$i-xMax$x/out-$c-$v-$w
                    echo "======================================"
                    echo -e "     \tcorpus   \tvecs\twins\titer\txMax"
                    echo -e "START\t$c\t$v\t$w\t$i\t$x\t($n/$t)"
                    echo "$outPath"
                    echo "--------------------------------------"

                    if [ ! -f $corpusFile ]; then
                        echo "File $corpusFile do not exists!"
                        exit 1
                    fi

                    if [ -d $outPath ]; then
                        echo "Path $outPath already exists!"
                    else
                        mkdir -p $outPath
                        ./glove.sh $corpusFile $outPath $v $w $i $x
                    fi
                done
            done
        done
    done
done

