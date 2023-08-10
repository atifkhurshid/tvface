import subprocess


def oscopy(filepath, dstdir):
    dstdir = dstdir + r'\\'
    cmd = ['xcopy', filepath, dstdir, '/K']
    proc = subprocess.run(cmd, capture_output=True)

    return proc
