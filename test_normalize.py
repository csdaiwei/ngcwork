
# csdaiwei@foxmail.com

import pdb

import re

with open('spec.0131.t0.s0.txt', 'r') as infile, open('spec.1031.t0.s0.norm.txt', 'w') as outfile:
	lines = infile.read().decode('utf-16').split('\n')
	
	# find initial speed (hss)
	m = re.search(r'HSS\t[0-9.]+', lines[1])
	if not m:
		outfile.write('error: hss not found\n')
		break;	#return?

	hss = float(m.group().split('\t')[1])

	# read data
	X, Y = [], []
	for ll in lines[5:]:
		ss = ll.split('\t')
		if len(ss)!= 2:
			continue
		x, y = float(ss[0]), float(ss[1])
		if x <= 15*hss:
			X.append(x)
			Y.append(y)
		else:
			break;

	# normalization
	ysum = sum(Y)
	X = [x/hss for x in X]
	Y = [y/ysum for y in Y]

	# save
	for (x, y) in zip(X, Y):
		outfile.write(str(x)+' '+str(y)+'\n')


