import requests
from urllib.parse import quote

def get_combined_emoji_image(emoji1, emoji2, size=128):
    base_url = "https://emk.vercel.app"
    emoji1_encoded = quote(emoji1)
    emoji2_encoded = quote(emoji2)
    full_url = f"{base_url}/s/{emoji1_encoded}_{emoji2_encoded}?size={size}"
    response = requests.get(full_url)
    if response.status_code == 200:
        return response.content
    else:
        return None
