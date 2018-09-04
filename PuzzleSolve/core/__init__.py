import tempfile
import shutil
from . import create

import os.path as path


class _Puzzle():
    
    def create(self, inFilename, outFilename, pieceLen):
        create.CreatePuzzle(inFilename, outFilename, pieceLen)

    def delete_temp(self):
        if hasattr(self, "tempfolder"):
            shutil.rmtree(self.tempfolder)

    def create_and_solve(self, inFilename, pieceLen):
        self.tempfolder = tempfile.mkdtemp()
        tempImage = path.join(self.tempfolder, path.basename(inFilename))
        self.create(inFilename, tempImage, pieceLen)


class Puzzle():
    def __enter__(self):
        self.puzzle_obj = _Puzzle()
        return self.puzzle_obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.puzzle_obj.delete_temp()
