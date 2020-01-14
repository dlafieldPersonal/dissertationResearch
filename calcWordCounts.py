with open("amazonRefined.txt", "r") as f:
	trans = f.readlines()
trans = trans[1:]
print("length of trans = " + str(len(trans)))

wdCts = [0]

for t in trans:
	x = t.split("\t")[0]
	l = len(x.split())
	while len(wdCts) < (l + 1):
		wdCts.append(0)
	wdCts[l] += 1

print("Length = " + str(len(wdCts)))
for i in range(len(wdCts)):
	if wdCts[i] > 0:
		print(str(i) + "\t" + str(wdCts[i]))
print("Done.")
