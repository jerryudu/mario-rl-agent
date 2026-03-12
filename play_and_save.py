import retro
import pygame
import numpy as np
import os
import gzip

# --- 設定區 ---
GAME = 'SuperMarioWorld-Snes'
STATE = 'YoshiIsland1'  # 你的起始關卡，如果是 Yoshi's Island 2 請改成對應名稱
OUTPUT_STATE_NAME = 'SecretDrill.state' # 存檔名稱
# -------------

def main():
    # 1. 初始化環境
    env = retro.make(game=GAME, state=STATE, render_mode='rgb_array')
    
    # 2. 初始化 Pygame (用來接收鍵盤輸入)
    pygame.init()
    screen_size = (256 * 3, 224 * 3) # 放大 3 倍比較好玩
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(f"Retro Player - Press 'S' to Save to {OUTPUT_STATE_NAME}")
    clock = pygame.time.Clock()

    # 3. 按鍵映射 (鍵盤 -> SNES 手把)
    #SNES Buttons: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
    # 對應索引:      0, 1,    2,     3,   4,    5,    6,     7, 8, 9, 10,11
    
    # 定義按鍵組合 (你可以依習慣修改)
    # 這裡設定：Z=跳(B), X=跑(Y), 方向鍵=移動, A=旋轉跳(A), Enter=Start
    key_mapping = {
        pygame.K_z: 0,      # B (跳)
        pygame.K_x: 1,      # Y (跑/拿)
        pygame.K_RSHIFT: 2, # Select
        pygame.K_RETURN: 3, # Start
        pygame.K_UP: 4,     # Up
        pygame.K_DOWN: 5,   # Down
        pygame.K_LEFT: 6,   # Left
        pygame.K_RIGHT: 7,  # Right
        pygame.K_a: 8,      # A (旋轉跳) - 這是重點！
        pygame.K_s: 9,      # X
        pygame.K_q: 10,     # L
        pygame.K_w: 11      # R
    }

    obs = env.reset()
    running = True

    print(f"\n🎮 開始遊戲！")
    print(f"👉 操作：方向鍵移動, Z=跳, X=跑, A=旋轉跳")
    print(f"💾 存檔：按鍵盤上的 '1' 鍵 (存檔並退出)")

    while running:
        # 限制 60 FPS
        clock.tick(60)

        # 處理 Pygame 事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 按下 '1' 鍵存檔
            if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                print("正在存檔...")
                # 取得二進位狀態數據
                state_data = env.em.get_state()
                
                # --- 【關鍵修改】使用 gzip 開啟並寫入 ---
                # Retro 讀取時預設認為這是 .gz 檔，所以我們必須壓縮它
                with gzip.open(OUTPUT_STATE_NAME, 'wb') as f:
                    f.write(state_data)
                # -------------------------------------
                    
                print(f"✅ 存檔成功！已儲存為: {OUTPUT_STATE_NAME}")
                print(f"現在你可以在訓練程式中將 STATE 改為 '{OUTPUT_STATE_NAME}' 了")
                running = False

        # 組合 Action
        # Retro 需要一個含有 12 個布林值/整數的陣列
        action = [0] * 12
        
        for key, btn_idx in key_mapping.items():
            if keys[key]:
                action[btn_idx] = 1

        # 修改後 (新版 Gymnasium 相容寫法)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 渲染畫面
        # Retro 給的是 (224, 256, 3) 的 RGB array
        # Pygame 需要 (Width, Height)，且通常是 Transpose 過的，這裡簡單做
        frame = np.transpose(obs, (1, 0, 2)) # 轉向
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.scale(surf, screen_size) # 放大
        
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            env.reset()

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()