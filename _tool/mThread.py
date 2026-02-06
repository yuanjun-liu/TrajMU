exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from multiprocessing.dummy import Lock as ThreadLock
from multiprocessing import Lock as ProcessLock
plock_default = ProcessLock()
tlock_default = ThreadLock()
