# GENERATE LINKS FROM YOUTUBE CHANNEL
import re
import requests
from bs4 import BeautifulSoup

# Function to extract video links from the given YouTube channel
def get_video_links_from_channel(channel_url):
    # Download the page using requests
    response = requests.get(channel_url)
    # If the response was successful, no Exception will be raised
    response.raise_for_status()
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all video links using regex that captures the full watch URL pattern
    video_links = re.findall(r'"(/watch\?v=[^"]+)"', str(soup))
    # Create full URLs from the video link fragments
    full_links = ['https://www.youtube.com' + link for link in video_links]
    # Remove duplicates by converting the list to a set and back to a list
    unique_links = list(set(full_links))
    return unique_links

# Example usage:
video_links = get_video_links_from_channel('https://www.youtube.com/@Lofigirl-Chillbeats/videos')

# GENERATE LINKS FROM PLAYLIST
from pytube import Playlist  
 
def get_video_links_from_playlist(playlist_url): 
    playlist = Playlist(playlist_url) 
    return [video.watch_url for video in playlist.videos]  
 

playlist_urls = ['https://www.youtube.com/watch?v=_zuVOdOiGlo&list=PLofht4PTcKYnaH8w5olJCI-wUVxuoMHqM&pp=iAQB']
''', 
                  'https://www.youtube.com/watch?v=DUpZt8Qg3fM&list=PLk_H7HxvSE2fTCUoWKG5YyfUwVFAAVnKL',
                  'https://www.youtube.com/watch?v=WTsmIbNku5g&list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo',
                  'https://www.youtube.com/watch?v=PFG3oSLjSv8&list=PLf-Sb-XKHsVfoal99nB45VsZVoCbkS479',
                  'https://www.youtube.com/watch?v=wGdodz6ck7g&list=PLi8ZVZZLpNVZitABWmrUKWq0lYNC_O3hw',
                  'https://www.youtube.com/watch?v=A7uNvvAKsYU&list=PLXIclLvfETS3AgCnZg4N6QqHu_T27XKIq',
                  'https://www.youtube.com/watch?v=v6YnNyAdybo&list=PL4gj7yxyrQ0_o8iSSaENWbnR9NuVdXOI2',
                  'https://www.youtube.com/watch?v=B9YBbB9-1cI&list=PLf-Sb-XKHsVcLUBoQRTNSXraxZP0gA9tD',
                  'https://www.youtube.com/watch?v=JNuSJvofw7A&list=PLXY_0YgPbMrPAcJdoN2yd_SNo9OsiIsNI',
                  'https://www.youtube.com/watch?v=jrTMMG0zJyI&list=PLOzDu-MXXLljgaHr8OsKj8X0mXlxaKhxS',
                  'https://www.youtube.com/watch?v=tYAyHsI3_Yc&list=PLbVVh-7RFMjqiYXkqHZvfu0898M4pNkwX',
                  'https://www.youtube.com/watch?v=TG3-pQOBfTk&list=PLEL3c81ok8OX_cvr8vr6s8wD2nm0kpMij',
                  'https://www.youtube.com/watch?v=bO2UgfjRVkk&list=PL6fhs6TSspZszor_x46PxP6USKFLV_Dze',
                  'https://www.youtube.com/watch?v=SFiwd1Lty94&list=PLXCoHsJ9oLedL4qvXh8uAooXpGuliDWiv',
                  'https://www.youtube.com/watch?v=lPCc78REQpU&list=PL6fhs6TSspZu4nYlvQ_l206FmRaMT_MGh',
                  'https://www.youtube.com/watch?v=zdYzL6wkr0A&list=PL_77ETNrRb7Ep0Zv3tQNLNxQgwTsTHNrV']'''

# Initialize an empty list to store all video URLs
all_video_urls = []

# Iterate over each playlist URL and get the video links
for playlist_url in playlist_urls:
    video_urls = get_video_links_from_playlist(playlist_url)
    all_video_urls.extend(video_urls)

print(len(all_video_urls))

# Remove duplicates by converting the list to a set and back to a list
all_video_urls = list(set(all_video_urls))

print(len(all_video_urls))

# Write the URLs to a text file
with open('music_youtube_urls.txt', 'w') as file:
    for url in all_video_urls:
        file.write(url + '\n')
