import numpy as np
import gymnasium as gym
import cv2  # <--- 優化三：改用 OpenCV 加速
from collections import deque
from stable_baselines3.common.monitor import Monitor
import retro

# --- 優化一：新增 MaxAndSkipEnv ---
class MaxAndSkipEnv(gym.Wrapper):
    """
    每 4 幀做一次動作 (skip=4)。
    返回最後 2 幀的 max (解決 SNES 閃爍問題)。
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # 取最後兩幀的最大值，避免物件閃爍消失
        if len(self._obs_buffer) == 2:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            max_frame = self._obs_buffer[0]

        return max_frame, total_reward, done, truncated, info

# --- 優化二：新增 StuckResetWrapper ---
class StuckResetWrapper(gym.Wrapper):
    """
    如果 X 座標在 n_steps 內都沒有進展，就強制重置 (視為失敗)。
    防止 AI 在原地發呆刷時間懲罰。
    """
    def __init__(self, env, max_stuck_steps=250):
        super().__init__(env)
        self.max_stuck_steps = max_stuck_steps
        self.stuck_counter = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.stuck_counter = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 檢查是否正在顯示訊息框 (從 info 讀取)
        is_reading_message = info.get("message_box", 0) > 0

        # 2. 卡死判定邏輯
        x_pos = info.get("x_pos", 0)
        
        if x_pos > self.max_x:
            self.max_x = x_pos
            self.stuck_counter = 0
            
        elif is_reading_message:
            # 【關鍵！】如果正在看訊息，暫停計時
            self.stuck_counter = 0 
            
        else:
            self.stuck_counter += 1
            
        if self.stuck_counter > self.max_stuck_steps:
            truncated = True
            info["stuck"] = True
            
        return obs, reward, terminated, truncated, info

# --- 優化三：改寫 PreprocessObsWrapper (用 OpenCV) ---
class PreprocessObsWrapper(gym.ObservationWrapper):
    """使用 OpenCV 進行 Resize 和 灰階化，比 Torchvision 快"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, 84, 84), dtype=np.float32
        )

    def observation(self, obs):
        # 1. 轉灰階 (如果是 RGB)
        if obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize 到 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 3. 增加 Channel 維度 (84, 84) -> (1, 84, 84)
        obs = np.expand_dims(obs, 0)
        
        # 4. Normalize 到 [-1, 1]
        obs = (obs / 255.0 - 0.5) / 0.5
        
        return obs.astype(np.float32)

class SimpleFrameStack(gym.Wrapper):
    """
    將最近 k 張畫面疊加在一起，讓 AI 能感知速度與方向。
    假設輸入是 (C, H, W)，堆疊 k 張後變成 (C*k, H, W)。
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        
        # 修改 Observation Space
        # 假設原本是 (1, 84, 84)，疊 4 張會變成 (4, 84, 84)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(shp[0] * n_stack, shp[1], shp[2]),
            dtype=env.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 一開始把第一張圖重複填滿 Buffer
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
        
    def _get_obs(self):
        # 將 deque 裡面的 frames 沿著 channel 維度接起來
        return np.concatenate(list(self.frames), axis=0)



class DiscreteActionWrapper(gym.ActionWrapper):
    """
    change action space from MultiBinary to Discrete with predefined button combos
    """
    def __init__(self, env, combos):
        super().__init__(env)


        if not hasattr(env.unwrapped, "buttons"):
            raise ValueError("unsupported env, must have 'buttons' attribute")

        self.buttons = list(env.unwrapped.buttons)  # e.g. ['B','Y','SELECT',...]
        self.button_to_idx = {b: i for i, b in enumerate(self.buttons)}

        # Get combos
        self.combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

        self._mapped = []
        n = env.action_space.n  # MultiBinary(n)
        for keys in combos:
            a = np.zeros(n, dtype=np.int8)
            for k in keys:
                if k not in self.button_to_idx:
                    raise ValueError(f"unsupported buttons in this env.buttons: {self.buttons}")
                a[self.button_to_idx[k]] = 1
            self._mapped.append(a)

    def action(self, act):
        return self._mapped[int(act)].copy()
    

class LifeTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def _get_lives(self, info):
        if not isinstance(info, dict):
            return None
        if "lives" in info:
            return int(info["lives"])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lives = self._get_lives(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = self._get_lives(info)

        died = False
        if lives is not None and self._prev_lives is not None:
            if lives < self._prev_lives:
                died = True
        self._prev_lives = lives

        if died:
            terminated = True
            if isinstance(info, dict):
                info = dict(info)
                info["death"] = True

        return obs, reward, terminated, truncated, info


class ExtraInfoWrapper(gym.Wrapper):
    """
    Attach extra RAM-derived signals (HUD timer, x-position) to info.
    """

    TIMER_HUNDREDS = 0x0F31
    TIMER_TENS = 0x0F32
    TIMER_ONES = 0x0F33
    # In SMW RAM, $0094 stores the low byte and $0095 stores the high byte.
    X_POS_LOW = 0x0094
    X_POS_HIGH = 0x0095
    # 【新增】金幣記憶體
    COINS_MEM = 0x0DBF

    # 【新增】Dragon Coin (耀西金幣) 記憶體位置
    # 在 Super Mario World (SNES) 中，這個位址紀錄當前關卡吃了幾顆 (0~5)
    DRAGON_COINS_MEM = 0x1420

    # 【補上這行！】關卡 ID 記憶體
    LEVEL_ID_MEM = 0x13BF

    # 【補上這一行！】
    POWERUP_STATUS = 0x0019  # 0=小隻, 1=大隻, 2=披風, 3=火球

    def __init__(self, env):
        super().__init__(env)
        self._episode_start_x = None

    def _get_ram(self):
        base_env = self.env.unwrapped
        if not hasattr(base_env, "get_ram"):
            return None
        return base_env.get_ram()

    # 【新增】讀取函式
    def _read_dragon_coins(self, ram):
        if ram is None: return 0
        return int(ram[self.DRAGON_COINS_MEM])

    # 【新增】讀取金幣函式
    def _read_coins(self, ram):
        if ram is None: return 0
        return int(ram[self.COINS_MEM])

    def _read_time_left(self, ram):
        if ram is None:
            return None
        hundreds = int(ram[self.TIMER_HUNDREDS]) & 0x0F
        tens = int(ram[self.TIMER_TENS]) & 0x0F
        ones = int(ram[self.TIMER_ONES]) & 0x0F
        return hundreds * 100 + tens * 10 + ones

    def _read_x_pos(self, ram):
        if ram is None:
            return None
        low = int(ram[self.X_POS_LOW])
        high = int(ram[self.X_POS_HIGH])
        return (high << 8) | low
    
    # 【新增這個函式】讀取變身狀態
    def _read_powerup(self, ram):
        if ram is None: return 0
        # 0=小隻, 1=大隻, 2=披風, 3=火球
        return int(ram[self.POWERUP_STATUS])

    def _inject_extra(self, info):
        ram = self._get_ram()
        time_left = self._read_time_left(ram)
        x_pos = self._read_x_pos(ram)
        
        # 【新增這段】讀取並寫入 info
        powerup = self._read_powerup(ram)
        
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        
        if time_left is not None: info["time_left"] = time_left
        if x_pos is not None:
            if self._episode_start_x is None: self._episode_start_x = x_pos
            info["x_pos"] = max(0, x_pos - self._episode_start_x)
            
        # 把變身狀態放入 info 字典，傳給下一個 wrapper
        if powerup is not None:
            info["powerup"] = powerup

        # 讀取金幣
        coins = self._read_coins(ram)
        if coins is not None:
            info["coins"] = coins

        # 【新增】讀取大金幣並寫入 info
        d_coins = self._read_dragon_coins(ram)
        if d_coins is not None:
            info["dragon_coins"] = d_coins # 記住這個 key，等下要用
            
        return info

    def reset(self, **kwargs):
        self._episode_start_x = None
        obs, info = self.env.reset(**kwargs)
        info = self._inject_extra(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_extra(info)
        return obs, reward, terminated, truncated, info


class AuxObservationWrapper(gym.Wrapper):
    """
    Convert image observations into a dict that also exposes scalar features (step/time).
    """

    def __init__(self, env, step_normalizer: float = 18000.0, time_normalizer: float = 300.0):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("AuxObservationWrapper expects a Box observation space as the image input")
        self.image_space = env.observation_space
        self.step_normalizer = max(step_normalizer, 1.0)
        self.time_normalizer = max(time_normalizer, 1.0)
        scalar_low = np.full((2,), -np.inf, dtype=np.float32)
        scalar_high = np.full((2,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": self.image_space,
                "scalars": gym.spaces.Box(low=scalar_low, high=scalar_high, dtype=np.float32),
            }
        )
        self._step_count = 0

    def _make_obs(self, obs, info):
        time_left = float(info.get("time_left", 0.0)) if isinstance(info, dict) else 0.0
        time_feat = np.clip(time_left / self.time_normalizer, 0.0, 1.0)
        step_feat = np.clip(self._step_count / self.step_normalizer, 0.0, 1.0)
        scalars = np.array([step_feat, time_feat], dtype=np.float32)
        return {"image": obs, "scalars": scalars}

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return self._make_obs(obs, info), reward, terminated, truncated, info


class RewardOverrideWrapper(gym.Wrapper):
    def __init__(self, env, win_reward: float = 400.0):
        super().__init__(env)
        self.win_reward = win_reward
        self._prev_score = None
        self._prev_x = None  # 新增：紀錄 X 座標
        self._start_level_id = None # 【新增】紀錄起點的關卡 ID
        self._prev_level_id = None

        # 【補上這行！】初始化變數
        self._prev_powerup = 0

        # 【補上這行！】初始化金幣紀錄變數
        self._prev_coins = 0

    def _reset_trackers(self, info):
        self._prev_score = info.get("score", 0)
        # 確保初始 x_pos 存在，若無則設為 0
        self._prev_x = info.get("x_pos", 0) 

        # 【補上這行！】重置時也要讀取
        self._prev_powerup = info.get("powerup", 0)

        # 【補上這行！】重置時也要讀取當前金幣
        self._prev_coins = info.get("coins", 0)

        # 【新增】重置時，記住「主地圖」的 ID 是多少
        current_id = info.get("level_id", 0)
        self._start_level_id = current_id
        self._prev_level_id = current_id

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_trackers(info)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        reward = 0.0
        
        # 1. 進度獎勵 (保持不變)
        current_x = info.get("x_pos", 0)
        if self._prev_x is not None:
            reward += (current_x - self._prev_x) * 1.0 
        self._prev_x = current_x

        # --- 【新增】金幣專屬獎勵 ---
        current_coins = info.get("coins", 0)
        if current_coins > self._prev_coins:
            # 只要吃到金幣，直接給 +50 分！
            # 這相當於往前跑了半個螢幕的距離，它絕對會回頭吃。
            reward += 100.0 
            print("💰 發財了！金幣 +1 (Reward +50)")
            
        self._prev_coins = current_coins
        # -------------------------

        # 2. 【已開啟】分數獎勵 (Score Reward)
        current_score = info.get("score", 0)
        if self._prev_score is not None:
            # 計算分數差值
            score_diff = current_score - self._prev_score
            
            # 如果分數增加了，給予獎勵
            if score_diff > 0:
                # 係數建議：0.01 ~ 0.1 之間
                # 原理：Super Mario World 一枚金幣通常是 100 分
                # 如果係數設 0.1 -> 吃到金幣 = +10.0 reward
                # 如果係數設 0.02 -> 吃到金幣 = +2.0 reward
                # 相比之下，移動 1 pixel 是 +1.0 reward
                # 你希望 AI 願意為了金幣「繞路」多遠？
                # 設 0.1 代表它願意為了金幣多走 10 pixel (或冒一點險)
                reward += score_diff * 0.5
                
        self._prev_score = current_score

        # --- 【核心修改】3. 變身獎勵 ---
        current_powerup = info.get("powerup", 0)
        
        # 如果現在狀態 > 上一次狀態 (例如 0變成1，代表吃到蘑菇)
        if current_powerup > self._prev_powerup:
            reward += 50.0  # 給予 50 分的大獎勵！(相當於往前跑 50 格)
            print("🍄 吃到蘑菇/變身！獎勵 +50")
            
        # 如果現在狀態 < 上一次狀態 (例如 1變成0，代表受傷)
        elif current_powerup < self._prev_powerup:
            reward -= 20.0  # 懲罰受傷，讓它學會保護蘑菇狀態
            
        self._prev_powerup = current_powerup
        # ----------------------------

        # 3. 死亡懲罰 (保持不變)
        if info.get("death", False):
            reward -= 50.0

        # 4. 通關獎勵 (保持不變)
        if terminated and not info.get("death", False):
             reward += self.win_reward
        
        # --- 【新增】密道探索獎勵 (Sub-level Reward) ---
        current_id = info.get("level_id", 0)
        
        # 條件 1: 關卡 ID 變了 (代表進了水管或上了天堂)
        # 條件 2: 變成的 ID 不是 0 (避免讀取錯誤)
        # 條件 3: ID 不等於初始主地圖 (代表是在「子區域」裡面)
        # 條件 4: 剛剛那一瞬間發生的 (Edge detection)
        if (current_id != self._prev_level_id) and \
           (current_id != self._start_level_id) and \
           (current_id != 0):
            
            # 給予超大獎勵！
            # +300 分相當於直接通關的分數，它會愛死這個水管
            reward += 300.0 
            print(f"🕳️ 發現新大陸！進入密道/子區域 (ID: {current_id}) 獎勵 +300")
            
        # 條件 5: 如果它在密道裡面待著，持續給一點點小甜頭，鼓勵它別急著出來
        # (可選，看你是否希望它在密道多探索一下)
        if current_id != self._start_level_id and current_id != 0:
            reward += 0.5 

        self._prev_level_id = current_id
        # -----------------------------------------------

        # 5. 時間懲罰 (建議稍微調低，給它一點時間去吃金幣)
        # 原本 -0.1，如果要吃金幣，建議改成 -0.05 或 -0.01
        # 這樣它才不會覺得「去吃金幣花的時間成本」太高而不去吃
        reward -= 0.02  # 每一步時間懲罰 

        if terminated or truncated:
            self._reset_trackers(info)

        # Normalize
        reward = reward / 10.0 

        return obs, reward, terminated, truncated, info

class InfoLogger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0 or terminated:
            print(info)
        return obs, reward, terminated, truncated, info

COMBOS = [
    #[],                  # 0: NOOP
    ["RIGHT"],           # 1: 走右
    ["LEFT"],            # 2: 走左（可選）
    ["DOWN"],            # 3: 下蹲
    ["B"],               # 4: 跳
    ["Y"],               # 5: 跑
    ["RIGHT", "B"],      # 6: 右 + 跳
    ["RIGHT", "Y"],      # 7: 右 + 跑
    ["RIGHT", "B", "Y"], # 8: 右 + 跳 + 跑
    ["LEFT", "B"],       # 10: 左 + 跳
    ["A"],            # 原地旋轉跳 (鑽地板前置動作)
    ["RIGHT", "A"],   # 旋轉跳 (進密道必備)
    #["LEFT", "Y"],       # 11: 左 + 跑
    #["LEFT", "B", "Y"],  # 12: 左 + 跳 + 跑
]

def make_base_env(game: str, state: str):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    
    # 1. 【新增】Frame Skipping (這要在所有處理之前)
    # 這會讓環境每 4 幀才回傳一次，大幅加速！
    env = MaxAndSkipEnv(env, skip=4)
    
    # 2. 縮圖轉灰階 (用 OpenCV 版)
    env = PreprocessObsWrapper(env)
    
    # 3. Frame Stacking
    env = SimpleFrameStack(env, n_stack=4)
    
    # 4. 動作空間
    env = DiscreteActionWrapper(env, COMBOS)
    
    # 5. 讀取 RAM 資訊
    env = ExtraInfoWrapper(env)
    
    # 6. 【新增】防止卡死
    env = StuckResetWrapper(env, max_stuck_steps=250)
    
    # 7. 死亡判定
    env = LifeTerminationWrapper(env)
    
    # 8. 獎勵重塑
    env = RewardOverrideWrapper(env)
    
    # 9. 字典觀測 (最後打包)
    env = AuxObservationWrapper(env)
    
    # 10. 監控 (放在 Aux 之後是為了能紀錄 scalar 等資訊，但要注意它記錄的是 shaped reward)
    env = Monitor(env)

    return env
