#!/bin/sh
echo "Copy data files"
cp data/ISTOLOGIE.csv data/ISTOLOGIE_corr.csv
cp data/RTRT_NEOPLASI.csv data/RTRT_NEOPLASI_corr.csv

echo "Remove CRLF"
sed -i ':a;N;$!ba;s/\r\n/ /g' data/ISTOLOGIE_corr.csv
sed -i ':a;N;$!ba;s/\r\n/ /g' data/RTRT_NEOPLASI_corr.csv

echo "Remove slash before quotes"
sed -i 's/\\\\[\\]*",/",/g' data/ISTOLOGIE_corr.csv
sed -i 's/\\\\[\\]*",/",/g' data/RTRT_NEOPLASI_corr.csv

echo "Remove quotes inside fields"
sed -i 's/\\"//g' data/ISTOLOGIE_corr.csv
sed -i 's/\\"//g' data/RTRT_NEOPLASI_corr.csv

#sed -i 's/\([^\\]\)\(\\"\)/\1/g' data/ISTOLOGIE_corr.csv
#sed -i 's/\([^\\]\)\(\\"\)/\1/g' data/ISTOLOGIE_corr.csv
#sed -i 's/¦\\"/¦/g' data/ISTOLOGIE_corr.csv

