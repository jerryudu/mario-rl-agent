import sys
import os
import gymnasium as gym

# 加入當前路徑，確保能 import 到你旁邊的檔案
sys.path.append(os.getcwd())

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ================= 重要：這裡要修改 =================
# TODO: 1. 請把 'YOUR_FILENAME' 改成你定義 CustomPPO 的那個檔案名稱 (不含 .py)
# 例如：如果 CustomPPO 在 hw8.py 裡，這裡就寫 from hw8 import CustomPPO, VisionBackbonePolicy
from YOUR_FILENAME import CustomPPO, VisionBackbonePolicy

# TODO: 2. 匯入原本的環境建立函式
# 如果 make_vec_env 也是在同一個檔案，就從那裡匯入。
# 如果是在 wrappers.py，就保留下面這行；如果不是，請修改。
from wrappers import make_base_env 
# ===================================================

# ================= 環境建立函式 (複製原本的邏輯) =================
# 為了確保環境跟原本一模一樣，我們重新定義一次
def _make_env_thunk(game: str, state: str):
    def _thunk():
        return make_base_env(game, state)
    return _thunk

def make_vec_env(game: str, state: str, n_envs: int, use_subproc: bool = True):
    env_fns = [_make_env_thunk(game, state) for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    return vec_env
# ==============================================================

# ================= 參數設定 (請填入你剛剛查到的值) =================
# TODO: 3. 填入變數值
GAME = "SuperMarioBros-v0"   # 請改成原本程式碼裡的 GAME 變數值
STATE = "Level1-1"           # 請改成原本程式碼裡的 STATE 變數值
N_ENVS = 8                   # 請改成原本程式碼裡的 N_ENVS 數值

# 舊模型路徑
OLD_MODEL_PATH = "logs/MyPPO_0/best_model.zip" 
# 總共要跑多少步
TOTAL_STEPS = 5000000
# ==============================================================

def train():
    # 1. 建立環境
    print(f"正在重建環境: Game={GAME}, State={STATE}, n_envs={N_ENVS}")
    train_env = make_vec_env(GAME, STATE, n_envs=N_ENVS)

    print(f"正在載入 CustomPPO 模型：{OLD_MODEL_PATH}")
    
    # 2. 載入模型
    # 注意：這裡改用 CustomPPO.load，而不是 PPO.load
    model = CustomPPO.load(OLD_MODEL_PATH, env=train_env)

    # 3. 設定自動存檔 Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path='./logs/checkpoints_continued/',
        name_prefix='ppo_continued'
    )

    print("開始繼續訓練 (500萬步)...")
    
    # 4. 開始訓練
    model.learn(
        total_timesteps=TOTAL_STEPS, 
        reset_num_timesteps=False,  # 關鍵：接續 TensorBoard 曲線
        tb_log_name="MyPPO",        # 保持原本的 Log 名稱
        callback=checkpoint_callback
    )

    model.save("ppo_final_5m_steps")
    print("訓練完成！")

if __name__ == "__main__":
    train()