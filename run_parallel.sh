#!/bin/bash

d=(
  100
  10
  250
  75
  100
  100
)
eta=(
  0.1
  0.01
  0.05
  0.1
  0.05
  0.2
)

for index in ${!d[*]}; do 
    python languageIdentification.py --window 5 -d ${d[$index]} -eta ${eta[$index]}
done
