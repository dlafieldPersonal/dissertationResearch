cp glove/build/vectors.txt glove.6B.50d.txt &&
cp glove/build/amazonRefined.txt . &&
cp glove/build/pureTranscripts.txt . &&
python getMissingGloveWords.py &&
echo Done
