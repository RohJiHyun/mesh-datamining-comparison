import os, signal

def abort_handler(signum, frame):
    print("abort ...", signum)





signal.signal(signal.SIGABRT, abort_handler)