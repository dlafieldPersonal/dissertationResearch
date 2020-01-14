import re

with open("amazonRefined.txt", "r") as f:
	trans = f.readlines()
i = 0
with open("processedAmazon.txt", "w") as f:
	for t in trans:
		i += 1
		if(i == 1 or i % 100 == 0):
			print(str(i) + " out of " + str(len(trans)))
		x = t.split("\t")[0]
		if False and i == 10000:
			break
		
		newX = ""
		for xx in x:
			isSpace = xx == ' '
			isAlphaCap = xx >= 'A' and xx <= 'Z'
			isAlphaLower = xx >= 'a' and xx <= 'z'
			isNum = xx >= '0' and xx <= '9'
			#isPunk = xx in ['.', ',', '?', '!', '$', '%', '*', '\'', '\"', '-', ':', '/', '+']
			isPunk = False
			
			#if isSpace or isAlphaCap or isAlphaLower or isNum:
			if isSpace or isAlphaCap or isAlphaLower or isNum or isPunk:
				newX += xx
			else:
				#print(str(xx) + "\t" + str(x))
				#print(xx, end=" ")
				if xx != '\'':
					newX += " "
			"""
			newX = re.sub('\.', ' . ', newX)
			newX = re.sub(',', ' , ', newX)
			newX = re.sub('!', ' ! ', newX)
			newX = re.sub('\?', ' ? ', newX)
			"""
			newX = re.sub(' +', ' ', newX)
		f.write(newX.rstrip().lstrip().upper() + "\t" + str(t.split("\t")[1]))
		
print("Done.") 
