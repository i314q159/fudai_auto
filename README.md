# 抖音直播间福袋自动蹲守

通过 adb 控制真机，识别直播间左上角福袋浮标，自动完成任务并等待开奖。**不绑定具体直播间**，自动跟随当前屏幕上的直播间。

## 依赖

- Python 3.10+（已在 3.13 验证）
- adb 已连接手机，`adb devices` 能看到设备
- Python 包见 `requirements.txt`

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
.venv\Scripts\python.exe fudai_auto.py --minutes 180 --interval 30
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--minutes` | 180 | 运行时长（分钟） |
| `--interval` | 30 | 基础巡检间隔（秒）；剩余时间久时会自动放慢 |
| `--room` | 空 | 可选：期望的房间标题关键字；留空则只跟随当前直播间 |
| `--serial` | IB8XCEM7IJBAMZY5 | adb 设备序列号 |

## 训练

除采集外的全部训练都走 `train.ps1`，一条命令跑完两个模型并打印产物：

```powershell
.\train.ps1                          # 检测模型 + 状态模型（约 7~10 秒）
.\train.ps1 -DryRun                  # 两个都只评估，不写任何文件
.\train.ps1 -SkipState               # 只训检测模型
.\train.ps1 -SkipDetector            # 只训状态模型
.\train.ps1 -Aug 16 -Epochs 800      # 状态模型增强/轮数
.\train.ps1 -Predict data\fudai\icons # 用状态模型预测图片/目录
.\train.ps1 -Watch -IntervalSec 300  # 样本数变化自动全量重训（常驻）
```

| 文件 | 用途 | 训练脚本 |
|---|---|---|
| `data\fudai\model\fudai_clf.npz` | `fudai_auto.py` 运行时检测福袋位置 | `train_fudai_clf.py` |
| `data\fudai\model\fudai_state.npz` | 3 分类状态（待参与/已参与/倒计时），只做分析 | `train_fudai.py` |

重训检测模型后需要重启 `fudai_auto.py` 才会加载新权重。

## 流程图

训练数据流与运行时加载：

![训练数据流与运行时加载](docs/pipeline.png)

运行时单轮循环：

![运行时单轮循环](docs/runtime_loop.png)

图片由 `docs/make_flowchart.py` 生成（PIL + 微软雅黑），流程变了重跑一次脚本即可更新。

## 工作机制

1. 每轮截图，先 OCR 直播间标题：切了直播间会记录 `直播间切换: A -> B` 并重置状态继续蹲。
2. 定位福袋浮标：优先用 `data/fudai/model/fudai_clf.npz`（HOG+颜色直方图线性分类器）在左上角区域滑窗打分，命中分 ≥0.12 即视为福袋；模型缺失时回退模板匹配 / OCR 倒计时反推。
3. 打开福袋面板：
   - 出现「一键发表评论」→ 点击完成福袋任务
   - 出现「加入购物粉丝团并关注主播」→ 点击加入粉丝团并关注，成功后同面板继续点「一键发表评论」，一次完成两步任务
   - 出现抖币/支付确认 → 不自动支付，记 `need_payment` 并退出面板
   - 显示「任务已完成 等待开奖」→ 已参与，关闭面板继续
4. 切换直播间后自动点一次主播「关注」（已关注则跳过）。
5. 读面板倒计时排开奖时间；临近开奖进 90 秒观察窗口，识别「中奖 / 未中奖」并记录。
6. 自适应节奏：剩余 >10 分钟每 3 分钟查一次，>4 分钟每 90 秒，临近开奖加密到 30 秒内。
7. 常亮保护：脚本外已设置 `screen_off_timeout` 最大、`stay_on_while_plugged_in`；截图失败会自动重连 adb、重启 adb server、退回 shell+pull。

## 目录与输出

```
fudai_auto.py        主脚本
fudai_feat.py        分类器特征（HOG+HSV）与滑窗打分，训练/推理共用
train_fudai_clf.py   检测模型训练 -> data/fudai/model/fudai_clf.npz
train_fudai.py       状态模型训练（3 分类 + LOO 评估）-> fudai_state.npz
train.ps1            一键训练/预测/常驻重训（封装上面两个 py）
bag_template.png     福袋图标模板（模型缺失时兜底）
requirements.txt     依赖
log/                 运行日志与截图
log/run_*.log        每次运行的文本日志
log/fudai_log.csv    参与/开关播/开奖结果记录
log/shots/           关键节点截图（同时作为检测模型负样本来源）
data/fudai/          福袋识别训练集（每识别到一次福袋裁一张图标样本）
data/fudai/icons/    图标样本 png（120x90，以福袋主体为中心）
data/fudai/labels.csv 样本标注（时间/直播间/标签/来源/bbox）
data/fudai/model/fudai_clf.npz    检测模型（运行时用，w/thresh/窗口尺寸）
data/fudai/model/fudai_state.npz  状态模型（分析用，W/b/mean/std/classes）
data/fudai/model/train_report.txt 检测模型训练报告
docs/make_flowchart.py            流程图生成脚本（PIL）
docs/pipeline.png                 训练数据流 / 模型加载图
docs/runtime_loop.png             运行时单轮循环图
.venv/               虚拟环境
```

样本采集规则：同一状态（待参与/已参与/倒计时）60 秒内只存一张，并对像素做去重；标签取图标下方 OCR 文字（待参与、已参与或倒计时）。

重新训练：`.venv\Scripts\python.exe train_fudai_clf.py`（会自动把 `log/shots` 里的截图当负样本，避开真实福袋框）。样本多了以后重训即可，主脚本重启后自动加载新模型。

## 注意

- 参与福袋可能要求发送直播间指定评论，脚本只点「一键发表评论」，不输入自定义文本。
- 「加入购物粉丝团并关注主播」属正常福袋任务，实测免费（无支付确认）；一旦弹出抖币/支付界面脚本不会确认，只在 `log/fudai_log.csv` 记 `need_payment`。
- 必须停留在直播间：开奖前离开会失去资格，脚本不会点关闭/返回。
- 单实例锁：脚本占用本地 59999 端口，重复启动会直接退出。
