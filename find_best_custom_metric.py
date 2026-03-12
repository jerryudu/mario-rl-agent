import os
import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import retro
from wrappers import make_base_env

# 如果你有定義 CustomPPO，請記得 import，或是直接把 PPO 改成 CustomPPO
# from train import CustomPPO 

# ==========================================
# 1. 設定區
# ==========================================
CHECKPOINT_DIR = "/home/jerry/Desktop/deepln/deepln8 (鑽水管版本)/hw8/hw8_sub/runs_smw/checkpoints"  # 你的 checkpoint 資料夾
GAME = 'SuperMarioWorld-Snes'
STATE = 'YoshiIsland1'   # 統一用第一關來考試
EVAL_EPISODES = 1        # 每個模型考幾次
MAX_STEPS = 4000         # 時間給長一點，讓它有機會跑遠

# ==========================================
# 2. 這裡必須包含你的 Wrapper 定義 
# (為了腳本能獨立運行，請把你的 wrappers.py 內容貼過來，或 import)
# ==========================================
# ... [請在這裡貼上你所有的 Wrapper class: MaxAndSkipEnv, ExtraInfoWrapper 等] ...
# ... [如果不貼，請確保下方 make_base_env 能正常運作] ...

# 假設你已經有 make_base_env 了 (或是從你的主程式 import)
# from wrappers import make_base_env 

def evaluate_with_custom_metric(model, env):
    """
    跑一次遊戲，並回傳 (Max X, Final Score, Coins, Custom Metric)
    """
    obs, info = env.reset()
    done = False
    steps = 0
    
    # 初始化追蹤變數
    max_x = 0
    final_score = 0
    final_coins = 0
    
    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- 數據擷取 ---
        # 1. 抓取 x_pos (取最大值)
        x_pos = info.get('x_pos', 0)
        if x_pos > max_x:
            max_x = x_pos
            
        # 2. 抓取 score (通常 info['score'] 是累計的)
        current_score = info.get('score', 0)
        if current_score > final_score:
            final_score = current_score
            
        # 3. 抓取 coins
        current_coins = info.get('coins', 0)
        if current_coins > final_coins:
            final_coins = current_coins
            
        done = terminated or truncated
        steps += 1

    # --- 計算你的自定義公式 ---
    # Formula: 0.01 * max_x_pos + 0.1 * score + 1 * coins
    custom_score = (0.01 * max_x) + (0.1 * final_score) + (1.0 * final_coins)
    
    return max_x, final_score, final_coins, custom_score

def main():
    # 搜尋所有 .zip 檔
    model_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    
    # 依照修改時間排序，從舊測到新
    model_files.sort(key=os.path.getmtime)

    if not model_files:
        print(f"❌ 找不到任何模型在: {CHECKPOINT_DIR}")
        return

    print(f"🔍 找到 {len(model_files)} 個模型，開始依據自定義公式評分...")
    print(f"🎯 公式: 0.01 * X_POS + 0.1 * SCORE + 1 * COINS")
    print("-" * 60)
    print(f"{'Model Name':<40} | {'Metric':<10} | {'X':<5} {'Scr':<6} {'Coin':<4}")
    print("-" * 60)

    best_model_name = ""
    best_metric_val = -float('inf')
    best_details = {}

    # 建立環境
    env = make_base_env(GAME, STATE)

    for i, model_path in enumerate(model_files):
        filename = os.path.basename(model_path)
        
        try:
            # 載入模型
            model = PPO.load(model_path, env=env)
            
            # 進行評估 (如果要更準，可以在外層加迴圈跑 3 次取平均)
            mx, scr, coin, metric = evaluate_with_custom_metric(model, env)
            
            # 印出結果
            print(f"[{i+1}/{len(model_files)}] {filename[:35]:<35} | {metric:8.2f} | {mx:<5} {scr:<6} {coin:<4}")

            # 比較分數
            if metric > best_metric_val:
                best_metric_val = metric
                best_model_name = filename
                best_details = {'x': mx, 'score': scr, 'coins': coin}
                print(f"   🔥 新霸主出現！(Score: {metric:.2f})")

        except Exception as e:
            print(f"❌ 讀取失敗 {filename}: {e}")

    env.close()

    print("\n" + "="*60)
    print("🏆 最終冠軍模型 🏆")
    print(f"檔案: {best_model_name}")
    print(f"總分: {best_metric_val:.2f}")
    print(f"細節: Max X-Pos: {best_details.get('x')} | Score: {best_details.get('score')} | Coins: {best_details.get('coins')}")
    print("="*60)

if __name__ == "__main__":
    main()