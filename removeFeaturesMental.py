listOfDiseases = ["mdd", "bpd", "sz", "psychosis"]
#listOfDiseases = ["mdd"]

for disease in listOfDiseases:
	print("Processing " + disease + "...")
	inputFileName = "transcriptsWithFeatures" + disease + ".csv"
	outputFileName = "transcripts" + disease + ".csv"
	
	with open(inputFileName, 'r') as f:
		lines = f.read().splitlines()

	with open(outputFileName, 'w') as f:
		for line in lines:
			splitLine = line.split("\t")
			f.write(splitLine[0].upper() + "\t" + splitLine[-1] + "\n")
print("Done.")
