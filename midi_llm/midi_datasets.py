'''
Download piano solos off of a webpage
recommended init params:
    website_url = "https://www.mfiles.co.uk/midi-original.htm"
    output_folder = "piano_midis"
    keyword = "piano"
'''

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class MIDI_Downloader:
    def __init__(self, url, output_folder, keyword):
        self.url = url
        self.output_folder = output_folder
        self.keyword = keyword

    def download_midi_files(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            midi_links = [a['href'] for a in soup.find_all('a', href=True) if
                          a.text and self.keyword.lower() in a.text.lower()]

            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            for midi_link in midi_links:
                absolute_url = urljoin(self.url, midi_link)
                file_name = os.path.join(self.output_folder, os.path.basename(absolute_url))
                with requests.get(absolute_url, stream=True) as response:
                    with open(file_name, 'wb') as midi_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            midi_file.write(chunk)

                print(f"Downloaded: {file_name}")

        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")