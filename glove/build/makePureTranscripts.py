print("Opening transcripts...")
with open("amazonRefined.txt", "r") as f:
	trans = f.readlines()
trans = trans[1:]
i = 0
with open("pureTranscripts.txt", "w") as f:
	for t in trans:
		i += 1
		if(i == 1 or i % 100 == 0):
			print(str(i) + " out of " + str(len(trans)))
		x = t.split("\t")[0]
		f.write(x + "\n")
print("Done.")
