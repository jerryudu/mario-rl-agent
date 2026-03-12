import retro

game = 'SuperMarioWorld-Snes'
print(f"正在查詢 {game} 的所有可用狀態...")

try:
    states = retro.data.list_states(game)
    print(f"找到 {len(states)} 個狀態檔：")
    for s in states:
        print(f" - {s}")
except Exception as e:
    print("查詢失敗，請確認遊戲名稱正確。")