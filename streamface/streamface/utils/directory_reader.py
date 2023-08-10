from pathlib import Path


class DirectoryReader(object):
    """Iterate through files in a directory"""

    def __init__(self, dir, exts, logger, resume=True):
        self.exts = exts
        self.logger = logger
        self.resume = resume
        
        self.dir = Path(dir)
        self.processed, self.lastfilename = self.getprocessed()
        self.filenames = self.getfilenames()


    def next(self, batch_size=1):
        if not self.filenames:
            self.filenames = self.getfilenames()
        
        if self.lastfilename:
            self.processed.write(f'{self.lastfilename}\n')
        
        filepaths = []
        for _ in range(min(batch_size, len(self.filenames))):
            filepath = self.filenames.pop(0)
            filepaths.append(filepath)

        if len(filepaths):
            self.lastfilename = filepaths[-1].name

        return filepaths


    def getfilenames(self):
        def is_extfile(fp):
            ret = True if fp.is_file() and fp.suffix in self.exts else False
            return ret

        filenames = sorted(
            [fp for fp in self.dir.iterdir() if is_extfile(fp)],
            key=lambda x: x.stem)

        if self.resume:
            unprocessed_filenames = filenames.copy()
            while unprocessed_filenames:
                if unprocessed_filenames.pop(0).stem in self.lastfilename:
                    return unprocessed_filenames
        else:
            self.resume = True

        return filenames


    def getprocessed(self):

        filepath = self.dir / 'processed.txt'

        if self.resume:
            filepath.touch(exist_ok=True)
            processed = open(filepath , 'r+')
        else:
            processed = open(filepath, 'w+')

        lastfilename = ''
        for line in processed:
            line = line.rstrip()
            if line:
                lastfilename = line

        return processed, lastfilename


    def close(self):

        if self.lastfilename:
            self.processed.write(f'{self.lastfilename}\n')

        self.processed.close()
