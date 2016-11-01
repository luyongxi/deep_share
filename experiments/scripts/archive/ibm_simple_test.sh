#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="../logs/ibm_simple_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../..
time ./tools/ibm_simple_test.py --gpu 0 \
  --model data/imdb_IBMAttributes/Ethnicity/ethnicitySecondRound.prototxt \
  --weights data/imdb_IBMAttributes/Ethnicity/ethnicitySecondRound.caffemodel \
  --folders Asian Black White

time ./tools/ibm_simple_test.py --gpu 0 \
  --model data/imdb_IBMAttributes/HairHatBald/hairhatbaldSecondRound.prototxt \
  --weights data/imdb_IBMAttributes/HairHatBald/hairhatbaldSecondRound.caffemodel \
  --folders Bald Hat Hair

time ./tools/ibm_simple_test.py --gpu 0 \
  --model data/imdb_IBMAttributes/HairColor/haircolor.prototxt \
  --weights data/imdb_IBMAttributes/HairColor/haircolor.caffemodel \
  --folders Blackhair Blondehair

time ./tools/ibm_simple_test.py --gpu 0 \
  --model data/imdb_IBMAttributes/FacialHair/facialhair.prototxt \
  --weights data/imdb_IBMAttributes/FacialHair/facialhair.caffemodel \
  --folders FacialHair NoFacialHair

time ./tools/ibm_simple_test.py --gpu 0 \
  --model data/imdb_IBMAttributes/Glasses/SunEyeNoGlasses.prototxt \
  --weights data/imdb_IBMAttributes/Glasses/SunEyeNoGlasses.caffemodel \
  --folders NoGlasses SunGlasses VisionGlasses