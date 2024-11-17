import requests
import pandas as pd

# URL của Data Dragon để lấy dữ liệu về các item (cập nhật với phiên bản mới nhất)
url = "https://ddragon.leagueoflegends.com/cdn/14.22.1/data/vi_VN/item.json"  # Thay đổi phiên bản nếu cần

# Gửi yêu cầu GET đến Data Dragon API để lấy danh sách các item
response = requests.get(url)
item_data = response.json()

# Lấy toàn bộ dữ liệu về các item
items = item_data['data']

# Tạo danh sách để lưu thông tin chi tiết về các item trong Summoner's Rift
items_list = []

# Lặp qua từng item để lấy thông tin chi tiết
for item_id, item_info in items.items():
    # Kiểm tra xem item có thể được mua trong Summoner's Rift không
    if 'maps' in item_info and '11' in item_info['maps'] and item_info['maps']['11']:
        # Thu thập thông tin chi tiết về item
        item_details = {
            'ID': item_id,
            'Name': item_info.get('name', 'N/A'),
            'Description': item_info.get('description', 'N/A'),
            'Gold Cost': item_info.get('gold', {}).get('total', 'N/A'),
            'Gold Sell': item_info.get('gold', {}).get('sell', 'N/A'),
            'Gold Purchase': item_info.get('gold', {}).get('purchasable', 'N/A'),
            'Tags': ', '.join(item_info.get('tags', [])),
            'Depth': item_info.get('depth', 'N/A'),
            'Stacks': item_info.get('stacks', 'N/A'),
            'Type': item_info.get('type', 'N/A'),
            'Group': item_info.get('group', 'N/A'),
            'Plaintext': item_info.get('plaintext', 'N/A'),
            'Required Champion': item_info.get('requiredChampion', 'N/A'),
            'Builds Into': ', '.join(item_info.get('into', [])),
            'Builds From': ', '.join(item_info.get('from', [])),
            'Effect': str(item_info.get('effect', {})),
            'Rarity': item_info.get('rarity', 'N/A'),
            'Special Recipe': item_info.get('specialRecipe', 'N/A'),
            'In Store': item_info.get('inStore', 'N/A'),
            'Item Lore': item_info.get('lore', 'N/A'),
            'Tooltip': item_info.get('tooltip', 'N/A')
        }
        
        # Thêm thông tin item vào danh sách
        items_list.append(item_details)

# Chuyển danh sách thành DataFrame của pandas
df = pd.DataFrame(items_list)

# Lưu dữ liệu vào file Excel
df.to_excel('summoners_rift_items.xlsx', index=False, engine='openpyxl')

print("Dữ liệu chi tiết của các item có thể mua trong Summoner's Rift đã được lưu vào file 'summoners_rift_items.xlsx'")
