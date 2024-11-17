import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from PIL import Image, ImageTk
import os

# Đọc dữ liệu từ file Excel
champions_df = pd.read_excel('Champion_Details_Ver2.xlsx')
items_df = pd.read_excel('Item_Details_Ver2.xlsx')

# Bảng ánh xạ mới với chỉ số chính và các chỉ số phụ
champion_item_mapping = {
    'Fighter': {'main': ['Damage', 'Health', 'LifeSteal', 'AbilityHaste'], 'secondary': ['AttackSpeed', 'Armor']},
    'Tank': {'main': ['Health', 'Armor', 'MagicResist'], 'secondary': ['HealthRegen', 'Tenacity', 'AbilityHaste']},
    'Support': {'main': ['CooldownReduction', 'ManaRegen', 'HealthRegen'], 'secondary': ['Active', 'Vision', 'AbilityHaste']},
    'Marksman': {'main': ['Damage', 'CriticalStrike', 'AttackSpeed', 'Onhit'], 'secondary': ['LifeSteal', 'ArmorPenetration']},
    'Mage': {'main': ['SpellDamage', 'Mana'], 'secondary': ['MagicPenetration','CooldownReduction', 'ManaRegen']},
    'Assassin': {'main': ['Damage', 'ArmorPenetration'], 'secondary': ['AbilityHaste', 'MovementSpeed']}
}

if 'Tags' not in items_df.columns:
    print("Cột 'Tags' không tồn tại trong file Items.xlsx")
else:
    # Lọc trang bị theo điều kiện "Mua được" là True
    items_df = items_df[(items_df['Gold Purchase'] == True)]
    boots = items_df[(items_df['Tags'].fillna('').str.contains("Boots")) & (items_df['Gold Cost'] > 500)]
    other_items = items_df[(~items_df['Tags'].fillna('').str.contains("Boots")) & (items_df['Gold Cost'] > 2500)]

    # Mã hóa cột Role và Tags
    le_role = LabelEncoder()
    champions_df['Role_encoded'] = le_role.fit_transform(champions_df['Tags'])

    le_type = LabelEncoder()
    items_df['Type_encoded'] = le_type.fit_transform(items_df['Tags'])

    # Kết hợp các đặc trưng để phân cụm KMeans
    item_features = items_df[['Type_encoded', 'Gold Cost']]

    # Áp dụng phương pháp Elbow để chọn số cụm tối ưu
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(item_features)
        distortions.append(kmeans.inertia_)

    # Sử dụng KMeans với số cụm tối ưu (chọn 5 cụm sau khi kiểm tra phương pháp Elbow)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    items_df['Cluster'] = kmeans.fit_predict(item_features)

    # Định nghĩa lớp DQN với Replay và Double DQN
    class DQN:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = []
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.model = self._build_model()
            self.target_model = self._build_model()

        def _build_model(self):
            model = models.Sequential()
            model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(layers.Dense(24, activation='relu'))
            model.add(layers.Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            return model

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) > 2000:
                self.memory.pop(0)

        def act(self, state):
            if np.random.rand() <= self.epsilon:
                return np.random.choice(self.action_size)
            act_values = self.model.predict(state)
            return np.argmax(act_values)

        def replay(self, batch_size):
            # Chỉ lấy mẫu nếu bộ nhớ đủ phần tử
            if len(self.memory) < batch_size:
                return

            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


        def update_target_model(self):
            self.target_model.set_weights(self.model.get_weights())

    
    def get_item_recommendation(champion_name, selected_role):
        # Lọc tướng từ danh sách tướng theo tên và loại tướng
        champion = champions_df[(champions_df['Name'] == champion_name) & (champions_df['Tags'].str.contains(selected_role))]
        if champion.empty:
            return ["Tướng không hợp lệ hoặc loại không phù hợp!"]

        # Lấy các tags phù hợp với loại tướng từ bảng ánh xạ
        role_mapping = champion_item_mapping.get(selected_role, {})
        main_tags = role_mapping.get('main', [])
        secondary_tags = role_mapping.get('secondary', [])

        # Điều kiện lọc các trang bị theo Tags
        if selected_role in ['Fighter']:
            excluded_tags = ['SpellDamage', 'MagicPenetration', 'OnHit', 'CriticalStrike']
        elif selected_role == "Assassin":
            excluded_tags = ['SpellDamage', 'MagicPenetration', 'SpeedAttack', 'OnHit']
        elif selected_role == 'Marksman':
            excluded_tags = ['SpellDamage', 'MagicPenetration', 'CooldownReduction', 'Health']
        elif selected_role == 'Tank':
            excluded_tags = ['Damage', 'LifeSteal', 'SpellDamage', 'CriticalStrike', 'MagicPenetration', 'ArmorPenetration', 'AttackSpeed', 'OnHit']
        elif selected_role == 'Mage':
            excluded_tags = ['Onhit', 'ArmorPenetration', 'LifeSteal', 'AttackSpeed', 'AbilityHaste']
        elif selected_role == 'Support':
            excluded_tags = ['Damage', 'LifeSteal', 'CriticalStrike', 'ArmorPenetration', 'AttackSpeed']
        else:
            excluded_tags = []

        # Lọc các trang bị chính và phụ theo các Tags cho từng loại tướng
        main_items = other_items[other_items['Tags'].apply(lambda tags: any(tag in tags for tag in main_tags))]
        secondary_items = other_items[other_items['Tags'].apply(lambda tags: any(tag in tags for tag in secondary_tags))]

        # Lọc các trang bị không thuộc các Tags bị loại bỏ
        main_items = main_items[~main_items['Tags'].apply(lambda tags: any(excluded_tag in tags for excluded_tag in excluded_tags))]
        secondary_items = secondary_items[~secondary_items['Tags'].apply(lambda tags: any(excluded_tag in tags for excluded_tag in excluded_tags))]

        # Lọc các giày theo từng loại tướng
        if selected_role == 'Fighter' or selected_role == 'Tank':
            boots_item = boots[boots['Tags'].apply(lambda tags: any(tag in tags for tag in ['Armor', 'SpellBlock']))]
        elif selected_role == 'Mage':
            boots_item = boots[boots['Tags'].apply(lambda tags: any(tag in tags for tag in ['MagicPenetration', 'CooldownReduction']))]
        elif selected_role == 'Marksman':
            boots_item = boots[boots['Tags'].apply(lambda tags: any(tag in tags for tag in ['Armor', 'SpellBlock', 'AttackSpeed']))]
        elif selected_role == 'Support':
            boots_item = boots[boots['Tags'].apply(lambda tags: any(tag in tags for tag in ['Armor', 'SpellBlock', 'MagicPenetration', 'CooldownReduction']))]
        else:
            boots_item = pd.DataFrame()  # Trường hợp không có loại tướng phù hợp

        # Chọn ngẫu nhiên một đôi giày
        if not boots_item.empty:
            boots_item = boots_item.sample(1)

        # Nếu có đủ trang bị chính và phụ, chọn ngẫu nhiên 4 trang bị chính, 1 trang bị phụ
        main_items = main_items.sample(3) if len(main_items) >= 4 else main_items
        secondary_items = secondary_items.sample(2) if len(secondary_items) >= 1 else secondary_items

        # Kết hợp lại các trang bị chính, phụ và giày
        final_items = pd.concat([main_items, secondary_items, boots_item])

        # Kiểm tra nếu có đủ 6 món trang bị khác nhau
        if len(final_items) < 6:
            # Nếu số lượng trang bị ít hơn 6, thêm các trang bị bổ sung
            remaining_items = other_items[~other_items['Name'].isin(final_items['Name'])]
            additional_items = remaining_items.sample(6 - len(final_items))
            final_items = pd.concat([final_items, additional_items])

        # Loại bỏ các trang bị trùng lặp
        final_items = final_items.drop_duplicates(subset=['Name'])

        # Tính reward dựa trên sự trùng khớp của Tags
        reward = 0
        for item_tags in final_items['Tags']:
            # Tính số lượng trùng khớp giữa các Tags của trang bị và các Tags chính/phụ
            matched_main = len([tag for tag in item_tags if tag in main_tags])
            matched_secondary = len([tag for tag in item_tags if tag in secondary_tags])
            reward += matched_main + matched_secondary  # Cộng vào reward

        # Sử dụng DQN để quyết định trang bị tốt nhất
        state = np.array([len(main_items), len(secondary_items), len(boots_item)])  # Ví dụ, sử dụng số lượng trang bị làm state
        action = dqn.act(state)  # Chọn hành động (ở đây sử dụng DQN để chọn trang bị tốt nhất)
        dqn.remember(state, action, reward, state, done=False)
        dqn.replay(32)  # Huấn luyện mô hình DQN với bộ nhớ hiện tại

        # Trả về tên của các trang bị đã chọn
        return final_items['Name'].tolist()

    
    # Khởi tạo mô hình DQN
    state_size = 3  # Chỉ có ba yếu tố (số lượng trang bị chính, phụ, và giày)
    action_size = 20  # Giả sử có 5 hành động cho việc chọn trang bị
    dqn = DQN(state_size, action_size)

    # Hàm xử lý khi thay đổi lựa chọn tướng
    def on_champion_select(event):
        champion_name = champion_combobox.get()
        champion = champions_df[champions_df['Name'] == champion_name]
        if not champion.empty:
            roles = champion.iloc[0]['Tags'].split(', ')
            role_combobox['values'] = roles
            role_combobox.set('')

    # Hàm xử lý nút bấm
    def on_recommend():
        champion_name = champion_combobox.get()
        selected_role = role_combobox.get()
        recommended_items = get_item_recommendation(champion_name, selected_role)
        
        # Hiển thị danh sách trang bị trong ô kết quả
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "\n".join(recommended_items))

    # Tạo giao diện Tkinter
    root = tk.Tk()
    root.title("Gợi ý trang bị cho tướng")

    # Combobox chọn tên tướng
    tk.Label(root, text="Chọn Tướng:").grid(row=0, column=0, padx=10, pady=5)
    champion_combobox = ttk.Combobox(root, values=champions_df['Name'].tolist())
    champion_combobox.grid(row=0, column=1, padx=10, pady=5)
    champion_combobox.bind("<<ComboboxSelected>>", on_champion_select)

    # Combobox chọn cách lên đồ (theo cột Loại của tướng)
    tk.Label(root, text="Chọn Loại Tướng:").grid(row=1, column=0, padx=10, pady=5)
    role_combobox = ttk.Combobox(root)
    role_combobox.grid(row=1, column=1, padx=10, pady=5)

    # Nút gợi ý
    recommend_button = tk.Button(root, text="Gợi ý Trang bị", command=on_recommend)
    recommend_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Textbox hiển thị kết quả gợi ý
    result_text = tk.Text(root, width=50, height=10)
    result_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

    root.mainloop()