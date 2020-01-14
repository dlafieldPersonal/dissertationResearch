#python preprocess.py &&
#cp processedAmazon.txt amazonRefined.txt &&
cp ../../justTranscripts.csv pureTranscripts.txt &&
#cp processedAmazon.txt amazonRefined.txt &&
#python makePureTranscripts.py &&
./vocab_count -verbose 2 -max-vocab 300000 -min-count 0 < pureTranscripts.txt > vocab.txt &&
./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < pureTranscripts.txt > cooccurrences.bin &&
./shuffle -verbose 2 -memory 8.0 < cooccurrences.bin > cooccurrence.shuf.bin &&
./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 50 -threads 8 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 0 -model 2 &&
echo finished

