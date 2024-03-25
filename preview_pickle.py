import pickle as pkl
import glob
from collections import Counter

pkl_files = glob.glob("battle_log/pokellmon_vs_bot/gpt-4-0125-preview_0.8_cot/*.pkl")
pkl_files.sort()

win_counter = Counter()

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        battle = pkl.load(f)
    if battle.player_username == 'yorhaha':
        win_counter[battle.won] += 1
    else:
        win_counter[not battle.won] += 1

print(win_counter)