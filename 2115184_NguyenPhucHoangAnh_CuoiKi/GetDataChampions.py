import requests
import pandas as pd

# URL của Data Dragon để lấy dữ liệu về các tướng (cập nhật với phiên bản trò chơi mới nhất)
url = "https://ddragon.leagueoflegends.com/cdn/14.22.1/data/vi_VN/champion.json"  # Thay đổi phiên bản nếu cần

# Gửi yêu cầu GET đến Data Dragon API để lấy danh sách các tướng
response = requests.get(url)
champions_data = response.json()

# Lấy danh sách tướng
champions = champions_data['data']

# Tạo danh sách để lưu thông tin chi tiết của từng tướng
champions_list = []

# Lặp qua từng tướng để lấy thông tin chi tiết
for champ_name, champ_info in champions.items():
    # URL để lấy chi tiết về từng tướng
    champ_detail_url = f"https://ddragon.leagueoflegends.com/cdn/14.22.1/data/vi_VN/champion/{champ_name}.json"
    champ_detail_response = requests.get(champ_detail_url)
    champ_detail_data = champ_detail_response.json()
    
    # Dữ liệu chi tiết của tướng
    champ_data = champ_detail_data['data'][champ_name]
    
    # Thu thập thông tin chi tiết nhất của tướng
    champ_info_dict = {
        'ID': champ_data.get('id', 'N/A'),
        'Name': champ_data.get('name', 'N/A'),
        'Title': champ_data.get('title', 'N/A'),
        'Blurb': champ_data.get('blurb', 'N/A'),
        'Tags': ', '.join(champ_data.get('tags', [])),
        'Partype': champ_data.get('partype', 'N/A'),
        'HP': champ_data['stats'].get('hp', 'N/A'),
        'HP Per Level': champ_data['stats'].get('hpperlevel', 'N/A'),
        'MP': champ_data['stats'].get('mp', 'N/A'),
        'MP Per Level': champ_data['stats'].get('mpperlevel', 'N/A'),
        'Attack Range': champ_data['stats'].get('attackrange', 'N/A'),
        'Move Speed': champ_data['stats'].get('movespeed', 'N/A'),
        'Attack Damage': champ_data['stats'].get('attackdamage', 'N/A'),
        'Attack Damage Per Level': champ_data['stats'].get('attackdamageperlevel', 'N/A'),
        'Armor': champ_data['stats'].get('armor', 'N/A'),
        'Armor Per Level': champ_data['stats'].get('armorperlevel', 'N/A'),
        'Spell Block': champ_data['stats'].get('spellblock', 'N/A'),
        'Spell Block Per Level': champ_data['stats'].get('spellblockperlevel', 'N/A'),
        'Attack Speed Offset': champ_data['stats'].get('attackspeedoffset', 'N/A'),
        'Attack Speed Per Level': champ_data['stats'].get('attackspeedperlevel', 'N/A'),
        'Passive Name': champ_data['passive'].get('name', 'N/A'),
        'Passive Description': champ_data['passive'].get('description', 'N/A'),
        'Passive Image': champ_data['passive']['image'].get('full', 'N/A')
    }
    
    # Lặp qua các kỹ năng (Q, W, E, R)
    for i, spell in enumerate(champ_data['spells']):
        champ_info_dict[f'Spell {i+1} Name'] = spell.get('name', 'N/A')
        champ_info_dict[f'Spell {i+1} Description'] = spell.get('description', 'N/A')
        champ_info_dict[f'Spell {i+1} Cooldown'] = spell.get('cooldownBurn', 'N/A')
        champ_info_dict[f'Spell {i+1} Cost'] = spell.get('costBurn', 'N/A')
        champ_info_dict[f'Spell {i+1} Range'] = spell.get('rangeBurn', 'N/A')
        champ_info_dict[f'Spell {i+1} Image'] = spell['image'].get('full', 'N/A')
    
    # Thêm thông tin vào danh sách
    champions_list.append(champ_info_dict)

# Chuyển danh sách thành DataFrame của pandas
df = pd.DataFrame(champions_list)

# Lưu dữ liệu vào file Excel
df.to_excel('detailed_champions_lol.xlsx', index=False, engine='openpyxl')

print("Dữ liệu chi tiết của các tướng đã được lưu vào file 'detailed_champions_lol.xlsx'")
