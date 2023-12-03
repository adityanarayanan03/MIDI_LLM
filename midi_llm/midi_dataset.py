import os
import fnmatch
from miditok import Structured, TSD, MIDILike, TokenizerConfig
from miditoolkit import MidiFile
from miditok.classes import TokSequence

import logging
logging.basicConfig()
logger = logging.getLogger("midi_dataset")
logger.setLevel(logging.DEBUG)

class MIDI_Dataset:
    def __init__(self, parent_dir, tokenization_method, verbose=0):
        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

        #Find all midi files within the parent directory
        self.files = []
        pattern = "*.mid"
        for root, dirs, files in os.walk(parent_dir):
            for filename in fnmatch.filter(files, pattern):
                self.files.append(os.path.join(root, filename))

        self.tokenization_method = tokenization_method

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

        # Defining tokenizer dictionary for selection
        tokenizer_dict = {
            'Structured': Structured(config),
            'TSD': TSD(config),
            'MIDILike': MIDILike(config)
        }

        self.tokenizer = tokenizer_dict.get(self.tokenization_method)

        for file in self.files:
            logger.debug(f"Attempting to tokenize file {file}")

            try:
                midi = MidiFile(file)

            except:
                logger.warning(f"Couldnt tokenize file {file}")
                continue

            this_file_tracks = self.tokenizer(midi)

            tracks += [x.ids for x in this_file_tracks]
        
        return tracks
    
    def write_midi_from_tokens(self, token_ids: list, filename: str):
        '''
        Writes input tokens into a midi file using the tokenizer
        MIDIDataset was instantated with.

        This is equivalent to detokenizing the input stream

        Returns the midi object and writes it to the input filename.
        '''
        ts = TokSequence()
        ts.ids = token_ids

        self.tokenizer.complete_sequence(ts)

        midi_out = self.tokenizer([ts])

        if ".mid" not in filename:
            filename += ".mid"

        midi_out.dump(filename)

        return midi_out

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