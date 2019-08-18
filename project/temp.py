

import os

files = os.listdir("../project/files")
files.remove(".DS_Store")
files.sort()
print(files)
for inx,item in  enumerate(files):
    print (inx,item)
