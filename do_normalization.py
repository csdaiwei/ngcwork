
# csdaiwei@foxmail.com

import pdb

import re
import os


inpathes = filter(lambda p:p.startswith('spec'), os.listdir('.')) 
inpathes = [p+'/' for p in inpathes]
outpathes = ['normalized/' + p for p in inpathes]

for outp in outpathes:
    if not os.path.exists(outp):
        os.makedirs(outp)

for inpath in inpathes:
    
    infiles = filter(lambda p:p.endswith('txt'), os.listdir(inpath)) 
    
    infiles = [inpath+t for t in infiles]
    outfiles =  ['normalized/'+t for t in infiles]

    for (infi, outfi) in zip(infiles, outfiles):

        with open(infi, 'r') as infile, open(outfi, 'w') as outfile:

            lines = infile.read().decode('utf-16').split('\n')
    
            # find initial speed (hss)
            m = re.search(r'HSS\t[0-9.]+', lines[1])
            if not m:
                outfile.write('error: hss not found\n')
                print infi+' hss error'
                continue;

            hss = float(m.group().split('\t')[1])

            # read data
            X, Y = [], []
            for ll in lines[5:]:
                ss = ll.split('\t')
                if len(ss)!= 2:
                    continue
                x, y = float(ss[0]), float(ss[1])
                if x < 15*hss:                         # do not need x >= 15
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

    print inpath+' processed'



    