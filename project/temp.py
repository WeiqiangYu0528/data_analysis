import json
import os



files = os.listdir("../static/files")
files.sort()
files.remove('.DS_Store')

print(files)
