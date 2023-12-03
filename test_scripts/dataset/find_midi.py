'''
Let's see if chatgpt can write working code...
'''
import os
import fnmatch

def find_files(directory, pattern='*.mid'):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            file_list.append(os.path.join(root, filename))
    return file_list

# Replace 'your_parent_directory' with the actual path to your parent directory
parent_directory = '../../bach/'
midi_files = find_files(parent_directory)

print("Found MIDI files:")
for midi_file in midi_files:
    print(midi_file)
