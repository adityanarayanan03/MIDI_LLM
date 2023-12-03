from midi_llm import MIDI_Dataset
from miditok import Structured, TSD, MIDILike, TokenizerConfig
from miditoolkit import MidiFile
from miditok.classes import TokSequence

config = TokenizerConfig()
tokenizer = Structured(config)

midi = MidiFile("../../bach/cellosui/01allema.mid")

tokenized = tokenizer(midi)
print(type(x := tokenized[0]))

ids = x.ids

print(type(ids))

ts = TokSequence()
ts.ids = ids

#print(f"Before, tok sequence is {ts}")
tokenizer.complete_sequence(ts)
#print(f"After, tok sequence is {ts}")

midi_out = tokenizer([ts])