# 🍄 Super Mario Bros RL Agent

![Demo Video/GIF Placeholder](vedio/mario_demo_small.gif)


## 📝 專案簡介 (Project Overview)
本專案使用**深度強化學習 (Deep Reinforcement Learning)** 來訓練 AI 代理人自動通關《超級瑪利歐兄弟》。
核心演算法基於 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 的 **PPO (Proximal Policy Optimization)**，並透過自訂神經網路架構 (`VisionBackbonePolicy`) 與客製化環境封裝 (`wrappers.py`) 來優化瑪利歐的學習效率與過關表現。

## 📂 專案架構 (Project Structure)
本專案將不同的訓練模組進行了解耦，確保程式碼具備良好的擴充性與可讀性：

- `Lab8 (從頭訓練).ipynb`: 完整訓練流程的 Jupyter Notebook，包含環境測試與初步模型訓練。
- `wrappers.py`: Gym 環境的客製化包裝，包含畫面裁切、灰階轉換、Frame Stacking (連續幀堆疊) 以及獎勵函數微調 (Reward Shaping)。
- `custom_policy.py`: 自定義的 CNN 策略神經網路架構，負責從遊戲畫面中萃取視覺特徵。
- `continue_train.py`: 接續訓練腳本，支援載入既有 Checkpoint 模型並透過多進程加速訓練。
- `eval.py`: 模型評估腳本，用來測試模型在特定關卡上的勝率與平均得分。
- `find_best_custom_metric.py`: 尋找與實驗最佳自訂評估指標 (Custom Metrics) 的腳本，用以優化訓練目標。
- `play_and_save.py`: 模型展示腳本，讓訓練好的瑪利歐進行遊戲，並將通關過程錄製成影片存檔。
- `vedio/`: 存放模型通關的展示影片檔。

## 🚀 環境安裝 (Installation)
請確保您的環境安裝了 Python 3.8+，接著安裝以下依賴套件：

```bash
pip install torch torchvision
pip install stable-baselines3[extra]
pip install gym-super-mario-bros
pip install opencv-python numpy
