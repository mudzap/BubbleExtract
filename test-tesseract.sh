#! /bin/bash

for f in ./*.png; do
  echo $f
  echo `tesseract -l jpn_vert $f $f`
done

