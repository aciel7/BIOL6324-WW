#!/bin/bash

if [ ! -e ./chp_extracted/ ]
then
cd ./raw_data/
apt-chp-to-txt --out-dir ../chp_extracted/ *.CHP
cd ../chp_extracted
rm *.log

for file in *
do
  sed -i "/#/d" $file
  sed -i "/#%GroupName*/d" $file
done
fi
