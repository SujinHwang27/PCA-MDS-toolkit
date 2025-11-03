#   If you see the error: ImportError: No module named 'matplotlib'
#   then you need to install matplotlib by doing: 
#   python -mpip install -U matplotlib

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if len(sys.argv) != 2 and len(sys.argv) != 3 :
    print('usage : ', sys.argv[0], 'file_2dpoints [file_labels]')
    sys.exit()

X = np.genfromtxt(sys.argv[1],delimiter=',',autostrip=True)
assert(X.shape[1] >= 2)  # X should be at least 2D. Take its first 2 columns
x1 = X[:, 0] # first column of X
x2 = X[:, 1] # second column of X

if(len(sys.argv) == 3) :
    y = np.genfromtxt(sys.argv[2], dtype='int')

fig, ax = plt.subplots()
ax.scatter(x1, x2)

if(len(sys.argv) == 3) :
    for i, txt in enumerate(y):
        ax.annotate(txt, (x1[i],x2[i]))

# save to pdf
plotname = sys.argv[1] + '_plot'
pdf = PdfPages(plotname + '.pdf')
pdf.savefig(fig)
pdf.close()
