import os
import time

if os.path.isfile("../output.txt"):
    os.rename("../output.txt", "../output%s.txt" % time.time())