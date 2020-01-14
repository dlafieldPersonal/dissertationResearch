import pickle as pk

quota = 100
shouldPrintAllUpdates = True

def ssqrt(al):
	(a, l) = al
	s = 0
	for ll in l:
		s += ll ** 2
	return s ** .5
	
def dProd(al, bl):
	(_, aa) = al
	(_, bb) = bl
	
	s = 0
	for i in range(len(aa)):
		s += aa[i] * bb[i]
	return s

def similarity(al, bl):
	return dProd(al, bl) / (ssqrt(al) * ssqrt(bl))
	
def applyListPos(lst, newCombo):
	(comboLst, maxSim, minSim) = lst
	(a, b, sim) = newCombo
	if len(comboLst) < quota:
		comboLst.append(newCombo)
		if sim > maxSim:
			maxSim = sim
		if sim < minSim:
			minSim = sim
		return (comboLst, maxSim, minSim)
	else:
		if sim > minSim:
			if sim > maxSim:
				maxSim = sim
			for i in range(len(comboLst)):
				(_, _, curSim) = comboLst[i]
				if curSim == minSim:
					comboLst[i] = newCombo
					(_, _, newMin) = comboLst[0]
					for j in range(1, len(comboLst)):
						(_, _, curMin) = comboLst[j]
						if curMin < newMin:
							newMin = curMin
					break
			if shouldPrintAllUpdates:
				print("Updated pos list:")
				print((comboLst, maxSim, newMin))
			return (comboLst, maxSim, newMin)
	return lst
	
def applyListNeg(lst, newCombo):
	(comboLst, maxSim, minSim) = lst
	(a, b, sim) = newCombo
	if len(comboLst) < quota:
		comboLst.append(newCombo)
		if sim > maxSim:
			maxSim = sim
		if sim < minSim:
			minSim = sim
		return (comboLst, maxSim, minSim)
	else:
		if sim < maxSim:
			if sim < minSim:
				minSim = sim
			for i in range(len(comboLst)):
				(_, _, curSim) = comboLst[i]
				if curSim == maxSim:
					comboLst[i] = newCombo
					(_, _, newMax) = comboLst[0]
					for j in range(1, len(comboLst)):
						(_, _, curMax) = comboLst[j]
						if curMax > newMax:
							newMax = curMax
					break
			if shouldPrintAllUpdates:
				print("Updated neg list:")
				print((comboLst, newMax, minSim))
			return (comboLst, newMax, minSim)
	return lst

def applyListZer(lst, newCombo):
	(comboLst, maxSim, minSim) = lst
	(a, b, sim) = newCombo
	if len(comboLst) < quota:
		comboLst.append(newCombo)
		if sim > maxSim:
			maxSim = sim
		if sim < minSim:
			minSim = sim
		return (comboLst, maxSim, minSim)
	else:
		if abs(sim) < abs(maxSim):
			if abs(sim) < abs(minSim):
				minSim = sim
			for i in range(len(comboLst)):
				(_, _, curSim) = comboLst[i]
				if abs(curSim) == abs(maxSim):
					comboLst[i] = newCombo
					(_, _, newMax) = comboLst[0]
					for j in range(1, len(comboLst)):
						(_, _, curMax) = comboLst[j]
						if abs(curMax) > abs(newMax):
							newMax = curMax
					break
			if shouldPrintAllUpdates:
				print("Updated neg list:")
				print((comboLst, newMax, minSim))
			return (comboLst, newMax, minSim)
	return lst

fileName = "glove.6B.50d.txt_backup"
#fileName = "smallGlove.txt"

print("reading glove file")
with open(fileName, "r") as f:
	gloveLst = f.readlines()
print("reading english words")
with open("englishWords.txt", "r") as f:
	#englishWordsList = f.readlines()[:-1]
	eee = f.readlines()[:-1]

englishWordsList = []
for i in range(len(eee)):
	eeee = eee[i][:-1]
	if eeee.isalpha():
		englishWordsList.append(eeee)
print("there are " + str(len(englishWordsList)) + " english words")	
print(englishWordsList[:10])
vecLst = []
print("filtering glove")
gCount = 0
for g in gloveLst:
	gCount += 1
	if gCount % 1000 == 0:
		print(str(gCount) + " out of " + str(len(gloveLst)))
	#ll = g.split()[1:]
	#print(ll)
	
	#exit()
	if g.split()[0].isalpha() and g.split()[0] in englishWordsList:
		vecLst.append((g.split()[0], list(map(float, g.split()[1:]))))
print("there are " + str(len(vecLst)) + " vectors")
with open("twitVecs.pkl", "wb") as f:
	pk.dump(vecLst, f)
h = int(len(vecLst))
vecLst = vecLst[h-1000:h+1000]

pos = ([], -999, 999)
neg = ([], -999, 999)
zer = ([], -999, 999)
count = 0

x = 0
minSim = 9999
maxSim = -9999
minZerSim = 9999
maxA = ""
maxB = ""
minA = ""
minB = ""
zerA = ""
zerB = ""
cartGoal = len(vecLst) - 1
cartGoal = int((cartGoal ** 2 + cartGoal) / 2)
count = 0
while x < len(vecLst) - 1:
	y = x + 1
	while y < len(vecLst):
		count += 1
		if count % 100000 == 0:
			print(str(count) + " out of " + str(cartGoal))
			
		a = vecLst[x]
		b = vecLst[y]
		sim = similarity(a, b)
		#print(a[0] + " " + b[0] + " " + str(similarity(a, b)))
		if sim > maxSim:
			maxSim = sim
			maxA = a[0]
			maxB = b[0]
		if sim < minSim:
			minSim = sim
			minA = a[0]
			minB = b[0]
		if abs(sim) < minZerSim:
			minZerSim = abs(sim)
			zerA = a[0]
			zerB = b[0]
		pos = applyListPos(pos, (a[0], b[0], sim))
		neg = applyListNeg(neg, (a[0], b[0], sim))
		zer = applyListZer(zer, (a[0], b[0], sim))
		y += 1
	x += 1
print(count)
print("\n\n\n")
print(maxA + " " + maxB + " " + str(maxSim))
print(minA + " " + minB + " " + str(minSim))
print(zerA + " " + zerB + " " + str(minZerSim))
print("")
print("pos:")
print(pos)
print("neg:")
print(neg)
print("zer:")
print(zer)

with open("similarListsTwitter.txt", "w") as f:
	for ccc in [(pos, "Most positive list:"),(neg, "Most negative list:"),(zer, "closest to zero list:")]:
		(lstWithStats, caption) = ccc
		f.write(caption + "\n")
		(lst, maxSim, minSim) = lstWithStats
		f.write("\tThe maximum similarity = " + str(maxSim) + "\n")
		f.write("\tThe minimum similarity = " + str(minSim) + "\n")
		for www in lst:
			f.write("\t" + str(www) + "\n")
		f.write("\n\n")
		
print("Done.")
