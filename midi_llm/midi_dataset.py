import os
import fnmatch
from miditok import Structured, TokenizerConfig
from miditoolkit import MidiFile

import logging
logging.basicConfig()
logger = logging.getLogger("midi_dataset")
logger.setLevel(logging.DEBUG)

class MIDI_Dataset:
    def __init__(self, parent_dir, verbose=0):
        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLeve(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

        #Find all midi files within the parent directory
        self.files = []
        pattern = "*.mid"
        for root, dirs, files in os.walk(parent_dir):
            for filename in fnmatch.filter(files, pattern):
                self.files.append(os.path.join(root, filename))
        
        self.tracks = self._tokenize_files()
        self._apply_token_mapping()
    
    def _apply_token_mapping(self):
        '''
        Applies a mapping onto self.tracks. This is only necessary becuase
        using the default token values results in some unknown characters when
        converted to pseudo-english using llama's *de*tokenizer.

        Because we're really trying to demonstrate in-context learning here,
        this mapping can be COMPLETELY arbitrary, as long as it avoids 
        unset token numbers in llama's token set
        '''
        logger.debug("Applying token mapping")
        for idx in range(len(self.tracks)):
            self.tracks[idx] = [x + 10000 for x in self.tracks[idx]]
    
    def _tokenize_files(self):
        '''
        Tokenizes all files in the dataset. Right now, it uses the 
        Structured tokenizer because that seems to give the easiest 
        tracks and results. We should come back and build a BPE tokenizer.

        Returns a list of token ids.
        '''
        tracks = []

        config = TokenizerConfig()
        tokenizer = Structured(config)

        for file in self.files:
            logger.debug(f"Attempting to tokenize file {file}")

            try:
                midi = MidiFile(file)

            except:
                logger.warning(f"Couldnt tokenize file {file}")
                continue

            this_file_tracks = tokenizer(midi)

            tracks += [x.ids for x in this_file_tracks]
        
        return tracks
    
    def __getitem__(self, index):
        return self.tracks[index]
    
    def __iter__(self):
        return iter(self.tracks)

if __name__ == "__main__":
    midi_dataset = MIDI_Dataset("bach/")

    print(midi_dataset[0])

    for idx, track in enumerate(midi_dataset):
        #logger.info(f"Track {idx} has {len(track)} tokens.")
        pass