import pygame
import time

class MIDI_Player:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

    def play_midi(self, file_path):
        try:
            pygame.mixer.music.load(file_path)
            print(f"Playing MIDI file: {file_path}")
            pygame.mixer.music.play()

            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(1)

        except pygame.error as e:
            print(f"Error playing MIDI file: {e}")

    def cleanup(self):
        pygame.mixer.quit()
        pygame.quit()

if __name__ == "__main__":
    # Replace 'your_midi_file.mid' with the path to your MIDI file
    midi_file_path = 'your_midi_file.mid'

    midi_player = MidiPlayer()
    midi_player.play_midi(midi_file_path)
    midi_player.cleanup()
