# 抖音直播间福袋自动蹲守

通过 adb 控制真机，识别直播间左上角福袋浮标，自动完成任务并等待开奖。**不绑定具体直播间**，自动跟随当前屏幕上的直播间。

## 依赖

- Python 3.11+（已在 3.13 验证；`config.toml` 用标准库 `tomllib`）
- adb 已连接手机，`adb devices` 能看到设备
- Python 包见 `requirements.txt`

```powershell
uv venv --python 3.13
uv pip install -r requirements.txt
```

## 运行

```powershell
.venv\Scripts\python.exe main.py --minutes 180 --interval 30
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--minutes` | 1440 | 运行时长（分钟），取自 `config.toml [run].minutes`（代码内置兜底 180） |
| `--interval` | 60 | 已跟到福袋、等开奖时的检查间隔（秒），取自 `[run].interval`；本房间还没跟到袋时用 `[detect].scan_interval_seconds` |
| `--room` | 空 | 可选：期望的房间标题关键字；留空则只跟随当前直播间（不一致只提示，不拦截） |
| `--serial` | 环境变量 `FUDAI_SERIAL`，未设置则取 `adb devices` 第一台 | adb 设备序列号 |
| `--cold-start-below` | 30 | 样本数低于该值时冷启动，取自 `[cold_start].below` |

命令行显式传参时覆盖 `config.toml`，`config.toml` 覆盖代码内置默认值。

## 配置 config.toml

检测节奏、开奖观察、采样节流都集中在根目录 `config.toml`，改完直接重启 `main.py` 生效（无需改代码）。时间单位统一为秒。

| 段 | 键 | 默认 | 说明 |
| --- | --- | --- | --- |
| `[run]` | `minutes` | 1440 | 运行时长（分钟） |
| | `interval` | 60 | 已跟到福袋、等开奖时的检查间隔 |
| `[room]` | `min_rounds` | 3 | 进入直播间时随机决定蹲几轮的下限（含端点） |
| | `max_rounds` | 9 | 随机轮数上限（含端点）；0 = 不切房 |
| | `switch_after_last_draw` | true | true 等最后一轮开奖结果出来再切；false 最后一轮点完立刻切 |
| | `swipe_attempts` | 3 | 上滑切房失败重试次数 |
| `[cold_start]` | `below` | 30 | 样本数低于该值走种子图 bootstrap |
| `[detect]` | `shot_retry_seconds` | 10 | 截图异常后的重试等待 |
| | `shot_error_log_every_seconds` | 30 | 连续截图异常的日志节流 |
| | `room_check_seconds` | 5 | 直播间标题巡房间隔（只认房间，不点福袋） |
| | `scan_interval_seconds` | 20 | 本房间还没跟到福袋时的巡检间隔（缩短以免错过短命福袋） |
| | `room_similarity` | 0.86 | 标题相似度≥该值视为同一房间，防 OCR 尾字抖动误判 |
| | `room_missing_warn_seconds` | 180 | 连续多久没标题就警告 |
| | `default_draw_seconds` | 300 | 读不到倒计时的默认排期（代码内置兜底 750） |
| | `new_cycle_jump_seconds` | 90 | 判断是不是同一个福袋：倒计时比预期回跳超过该值、或上一个袋已过开奖时刻，才算新袋；读不到倒计时时改用「距上次点击是否超过该值」判定 |
| | `watch_seconds` | 90 | 开奖观察窗口时长 |
| | `result_auto_close_seconds` | 12 | 中奖/未中奖卡点不掉时，等它自动关闭的最长时间（不按返回键） |
| | `draw_margin_seconds` | 3 | 提前多少秒进观察窗口 |
| | `tap_wait_seconds` | 6 | 点击后等界面反应的最长时间（界面一变就继续；至少看 2 次识别才放弃） |
| | `panel_open_settle_seconds` | 3 | 点开福袋后先等面板渲染，再截第一张识别图 |
| | `panel_settle_seconds` | 0.5 | 面板处理后的隔拍 |
| `[pacing]` | `min_sleep_seconds` | 1.5 | 每轮最短休眠（检查间隔读 `[run].interval` / `[detect].scan_interval_seconds`） |
| `[sample]` | `throttle_seconds` | 60 | 同状态样本采集节流 |
| `[watch]` | `poll_seconds` | 3 | 开奖窗口内截图/识别间隔 |
| | `log_every_seconds` | 20 | “等待开奖中”日志节流 |

`config.toml` 语法错误会在启动时打印 `[config]` 提示并使用内置默认值；未知段/未知键打印提示后忽略；值类型按内置默认值校验，写错（例如把开关写成字符串 `"false"`）会打印提示并沿用默认值。

巡房与福袋检测是分开的：每 `room_check_seconds`（默认 5 秒）截图 + OCR 标题。标题变化视为重启：重置轮次/开奖时间并立即重新检测福袋，换房后也要等新房间确认有福袋才点关注；点福袋仍按 `interval` / `scan_interval_seconds` 和 `[pacing]` 的节奏，不会因为巡房变快而每 5 秒点一次面板。

## 训练

除采集外的全部训练都走 `train.ps1`，一条命令跑完两个模型并打印产物：

```powershell
.\train.ps1                          # 样本 ≥30 全量训练；<30 自动冷启动（种子图 bootstrap）
.\train.ps1 -ColdStartBelow 20       # 自定义冷启动阈值
.\train.ps1 -Seed data\fudai\seed\seed_vgn.png  # 指定种子图
.\train.ps1 -DryRun                  # 两个都只评估，不写任何文件
.\train.ps1 -SkipState               # 只训检测模型
.\train.ps1 -SkipDetector            # 只训状态模型
.\train.ps1 -Aug 16 -Epochs 800      # 状态模型增强/轮数
.\train.ps1 -Predict data\fudai\icons # 用状态模型预测图片/目录
.\train.ps1 -Watch -IntervalSec 300  # 样本数变化自动全量重训（常驻）
```

冷启动逻辑：`data/fudai/icons` 里的样本数 < `-ColdStartBelow`（默认 30）时，检测模型改为用一张种子图 bootstrap（`python -m bootstrap_fudai_clf`），只做“点火”；样本涨到阈值后再跑全量训练。bootstrap 的阈值搜索同样不低于 `fudai_feat.MIN_THRESH`，自评 F1 < 0.5 会直接失败退出（不写模型、不 `--install`），避免把不可用的点火模型装成生产模型。状态模型要求至少 2 个类别，冷启动阶段会自动跳过。`main.py` 启动时读 `config.toml [cold_start].below`（默认 30）做同样判断：没有检测模型且样本不足，会先 bootstrap 再热加载。`train.ps1` 的 `-ColdStartBelow` 是独立参数，改阈值时两边保持一致。

| 文件 | 用途 | 训练脚本 |
| --- | --- | --- |
| `data\fudai\model\fudai_clf.npz` | 运行时检测福袋位置（`w/thresh/win_w/win_h`） | `python -m train_fudai_clf` 或 `python -m bootstrap_fudai_clf` |
| `data\fudai\model\fudai_state.npz` | 3 分类状态（待参与/已参与/倒计时），只做分析 | `python -m train_fudai` |

检测模型训练按 7:1.5:1.5 划分 训练/验证/holdout：正则强度与阈值只在验证集上选，报告里 `holdout` 那一行才是真实指标（`val` 行是选阈值用的，偏乐观）；阈值下限与运行时一致。

重训后下次启动 `main.py` 时加载新权重；冷启动路径会当场调用 `reload_clf()` 热加载。

## 换机复现

仓库里只放代码、模板图和文档。`data/fudai/icons`、`labels.csv`、`seed/`、`model/` 都在 `.gitignore` 里，**本机训练产物不提交**，所以换机后有两种恢复方式：

```powershell
git clone git@github.com:i314q159/fudai_auto.git
cd fudai_auto
uv venv --python 3.13
uv pip install -r requirements.txt
adb devices                       # 确认设备已授权
```

- **只要流程能跑**：直接启动 `main.py`。检测模型缺失且样本数低于 `[cold_start].below` 时，用 `data/fudai/seed/seed_vgn.png` 做 bootstrap；没有种子图会在日志里写明缺失路径并继续用模板匹配兜底。
- **要接着原来的数据练**：把旧机器的整个 `data/fudai/` 拷过来（icons、labels.csv、seed、model 一起），再按需跑 `.\train.ps1`。
- 手机换了：`--serial` 或环境变量 `FUDAI_SERIAL` 覆盖；ROI 与点击坐标统一按 1080×2400 设计，运行时按实际截图宽高自动等比缩放（`SCALE` + `sc_roi()`/`sc_pt()`），换分辨率不用改代码。可用 `python tests\check_resolution.py <截图>` 离线验证多种尺寸下的福袋定位、标题、倒计时与开奖文案识别。

## 流程图

采集 → 训练（冷启动 / 全量）→ 模型装配：

![采集、训练与装配](docs/pipeline.png)

`main.py` 入口与单轮巡检循环：

![运行时入口与单轮循环](docs/runtime_loop.png)

图片由 `docs/make_flowchart.py` 生成（PIL + 微软雅黑），流程变了重跑一次脚本即可更新。

## 工作机制

1. 每轮截图，先 OCR 直播间标题：**确认本房间有福袋后**才点一次主播「关注」（没检测到福袋的房间不点关注）；标题变化视为重启，重置轮次/开奖时间并立即重新检测福袋，新房间同样要等确认有福袋才关注；标题不变则按预定节奏继续蹲守。连续超过阈值无标题会警告并跳过福袋检查。
2. 面板已打开则直接处理任务；否则在屏幕上部**整宽的一条检测带**（设计坐标 y 270~480，横向铺满整屏，袋子被券包/榜单挤到右侧也能覆盖）里定位福袋浮标：优先用 `fudai_clf.npz`（HOG+颜色直方图线性分类器）滑窗打分，命中分达到阈值即视为福袋（阈值下限 0.12 由 `fudai_feat.MIN_THRESH` 统一约束训练与运行时，模型阈值低于它时按 0.12 用）；模型缺失/维度不符/打分失败回退模板匹配、OCR 倒计时反推。
3. 打开福袋面板后最多连点 3 步，每步都重新截图再判定（点完等底部文字变化就继续，最长 `tap_wait_seconds`；打开福袋也改成等面板出现即继续，不再固定 sleep）：
   - 出现「一键发表评论」→ 点击完成福袋任务
   - 出现「加入购物粉丝团并关注主播」→ 点击加入粉丝团并关注，成功后同面板继续点「一键发表评论」，一次完成两步任务
   - 出现抖币/支付确认 → 不自动支付，记 `need_payment` 并停手
   - 显示「任务已完成 等待开奖」→ 已参与，关闭面板继续
4. 只有点开后面板确认打开（识别到参与任务或「已参与」文案）才裁 120×90 图标入训练集，防止误检（券图标、头像等）污染数据；只点到图标但面板没打开不入库。
5. 读面板倒计时排开奖时间；**只有点到「一键发表评论」才算一轮**——只加粉丝团、只点兜底确认按钮、只在面板看剩余时间、重复点同一个福袋都不计轮次（新袋判定：倒计时比预期回跳超过 `new_cycle_jump_seconds`，或上一个袋已过开奖时刻；倒计时读不到时按距上次点击的时长兜底）；临近开奖进 90 秒观察窗口，识别「中奖 / 未中奖」。**中奖卡和未中奖卡用同一套关闭逻辑**：文字按钮（我知道了/确定/开心收下）→ 卡片关闭 ×（约 `(540,1704)`）→ 等它自己消失（两种卡都会定时自动关闭，**不按返回键**，避免退出直播间），关掉后立即回到福袋检测；中奖卡的「填写地址 领取奖品」不点，避免跳去领奖页。等满 `result_auto_close_seconds` 还没关时本轮跳过福袋检查、下一轮再试，不在弹层上继续点。
6. 每个直播间蹲满「进入时随机定下的 3~9 轮」（`[room].min_rounds` ~ `[room].max_rounds`，含端点）后上滑切下一间：默认等最后一轮开奖结果出来再切（`switch_after_last_draw = true`），切房成功后轮次/开奖状态清零，新房间要等确认有福袋才点关注；上滑失败按 `swipe_attempts` 重试，全失败则留在原房间重新计轮并重新抽一次目标轮数。
7. 巡检节奏分两档：**已跟到福袋、等开奖**时每 `[run].interval`（默认 60 秒）检查一次剩余时间；**本房间还没跟到袋**时按 `[detect].scan_interval_seconds`（默认 20 秒）更快巡，避免错过生存时间短的福袋。开奖窗口仍按 `next_draw_ts` 提前 `draw_margin_seconds` 进入，不会漏掉开奖时刻。
8. 常亮保护：脚本外已设置 `screen_off_timeout` 最大、`stay_on_while_plugged_in`；截图失败会自动重连 adb、重启 adb server、退回 shell+pull。
9. 分辨率自适应：所有 ROI / 点击点以 1080×2400 为设计基准，`shot()` 每次截图后刷新 `SCALE(x=W/1080, y=H/2400)`；分类器滑窗、模板匹配、样本裁剪、上滑手势、弹层关闭位都按同一系数换算，启动日志会打印当前屏幕尺寸与缩放比。

## 目录与输出

```text
main.py              运行入口：CLI / 单实例锁 / 冷启动判断 / 主循环
human/adb_conn.py    探测 adb 设备序列号（未设置 FUDAI_SERIAL 时取第一台已授权设备；没有 adb/设备返回 None）
fudai_auto/          运行时能力包（对外仍是 import fudai_auto as fa，API 不变）
  __init__.py        统一导出 + 可变状态转发（fa.SERIAL / fa.RUN_LOG / fa.ROOM_NAME）
  state.py           可变状态：设备序列号 / 日志路径 / 房间名 / 检测模型
  paths.py           目录与文件路径
  config.py          config.toml 读取与默认值
  logbook.py         运行日志 / 事件 CSV / 截图
  layout.py          1080×2400 设计坐标 ROI、弹层关键词、分辨率缩放
  device.py          adb 自愈、截图、点击、上滑
  ocrkit.py          OCR 引擎与倒计时解析
  detector.py        福袋定位：分类模型 / 模板匹配 / OCR 反推
  room.py            直播间标题与关注主播
  panel.py           福袋面板状态、任务点击、样本入库
  result.py          开奖弹层识别 / 统一关闭 / 开奖观察窗口
  dataset.py         福袋样本采集
fudai_feat/          特征包：HOG+HSV 特征与滑窗打分，训练/推理共用
  params.py          窗口/特征尺寸与判定阈值下限常量
  descriptor.py      单窗口 -> 特征向量
  sliding.py         ROI 滑窗打分
train_fudai_clf/     检测模型训练包（python -m train_fudai_clf）
  paths.py           路径与设计基准
  samples.py         正/负样本构造
  train.py           线性判别 + 阈值搜索 + 落盘
  __main__.py        命令行入口
bootstrap_fudai_clf/ 冷启动包（python -m bootstrap_fudai_clf）
  paths.py           模型路径与设计基准
  seed.py            种子图读取、福袋主体定位
  samples.py         单图正/负样本构造
  train.py           强正则判别 + 同帧自评阈值
  __main__.py        命令行入口
train_fudai/         状态模型训练包（python -m train_fudai，3 分类 + LOO 评估）
  paths.py           路径与类别常量
  dataset.py         样本装载、标签归一化、统计、稳定随机种子
  features.py        HOG + HSV 特征、数据增强
  model.py           softmax 分类器与模型存取
  eval.py            留一交叉验证
  cli.py             训练 / 预测流程与参数
  __main__.py        命令行入口
train.ps1            一键训练/预测/常驻重训（含冷启动阈值）
data/fudai/bag_template.png  福袋图标模板（模型缺失时兜底，已入库）
requirements.txt     依赖
config.toml          检测节奏/开奖观察/采样节流配置（CLI 可覆盖）
log/                 运行日志与截图
log/run_*.log        每次运行的文本日志
log/fudai_log.csv    参与/标题变化/开奖结果记录
log/shots/           关键节点截图（同时作为检测模型负样本来源）
data/fudai/          福袋识别训练集（本机产物，不入库）
data/fudai/icons/    图标样本 png（120x90，以福袋主体为中心）
data/fudai/labels.csv 样本标注（时间/直播间/标签/来源/bbox）
data/fudai/seed/     冷启动种子图（不入库）
data/fudai/model/    检测/状态模型与训练报告（不入库）
docs/make_flowchart.py            流程图生成脚本（PIL）
docs/pipeline.png                 采集 / 训练 / 装配图
docs/runtime_loop.png             运行时入口与循环图
tests/check_resolution.py         离线分辨率自检（缩放假截图跑检测链路，不连手机）
.venv/               虚拟环境
```

样本采集规则：确认面板打开后才入库；同一标签 60 秒内只存一张，且与上一张像素差 <2.0 时跳过；标签取图标区 OCR 的「待参与/已参与/倒计时 mm:ss」，都读不到时记 `icon_only`。

重新训练优先用 `.\train.ps1`（自动汇总样本与负样本、生成报告）；脚本内部等价于分别跑 `python -m train_fudai_clf` 和 `python -m train_fudai`。样本多了以后重训即可，重启 `main.py` 自动加载新模型。

## 注意

- 参与福袋可能要求发送直播间指定评论，脚本只点「一键发表评论」，不输入自定义文本。
- 「加入购物粉丝团并关注主播」属正常福袋任务，实测免费（无支付确认）；一旦弹出抖币/支付界面脚本不会确认，只在 `log/fudai_log.csv` 记 `need_payment`。
- 必须停留在直播间：开奖前离开会失去资格，脚本不会点关闭/返回。
- 单实例锁：脚本占用本地 59999 端口。重复启动时，新进程会自动结束旧实例并接管；如果端口被其它程序占用则拒绝启动。
