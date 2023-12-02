import pygame
import time

'''
How to use: 
    midi_player = MIDI_Player()
    midi_player.play_midi(midi_file_path)
    midi_player.cleanup()
'''


class MIDI_Player:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

    def play_midi(self, file_path):
        try:
            pygame.mixer.music.load(file_path)
            print(f"Playing MIDI file: {file_path}")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)

        except pygame.error as e:
            print(f"Error playing MIDI file: {e}")

    def cleanup(self):
        pygame.mixer.quit()
        pygame.quit()
