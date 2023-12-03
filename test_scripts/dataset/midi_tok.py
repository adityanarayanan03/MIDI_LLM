#Perhaps an attempt at learning a vocabulary from the bach midi dataset.
from miditok import Structured, TokenizerConfig
from miditoolkit import MidiFile
from pathlib import Path

# Creating a multitrack tokenizer configuration, read the doc to explore other parameters
config = TokenizerConfig()
tokenizer = Structured(config)

# Loads a midi, converts to tokens, and back to a MIDI
midi = MidiFile('../../bach/suites/airgstr4.mid')
tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens


print(f"This midi file has {len(tokens)} tracks.")

for idx, track in enumerate(tokens):
    print(f"Track {idx} has {len(track.ids)} tokens.")


""" converted_back_midi = tokenizer(tokens)  # PyTorch / Tensorflow / Numpy tensors supported

# Tokenize a whole dataset and save it at Json files
midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))
data_augmentation_offsets = [2, 1, 1]  # data augmentation on 2 pitch octaves, 1 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path("path", "to", "tokens_noBPE"),
                                data_augment_offsets=data_augmentation_offsets)

# Constructs the vocabulary with BPE, from the token files
tokenizer.learn_bpe(
    vocab_size=10000,
    tokens_paths=list(Path("path", "to", "tokens_noBPE").glob("**/*.json")),
    start_from_empty_voc=False,
)

# Saving our tokenizer, to retrieve it back later with the load_params method
tokenizer.save_params(Path("path", "to", "save", "tokenizer.json"))
# And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
tokenizer.push_to_hub("username/model-name", private=True, token="your_hugging_face_token")

# Applies BPE to the previous tokens
tokenizer.apply_bpe_to_dataset(Path('path', 'to', 'tokens_noBPE'), Path('path', 'to', 'tokens_BPE')) """