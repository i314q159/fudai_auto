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

## 工作机制

1. 每轮截图，先 OCR 直播间标题：切了直播间会记录 `直播间切换: A -> B` 并重置状态继续蹲。
2. 定位福袋浮标：模板匹配优先，分数不够时用 OCR 倒计时标签反推坐标（避免粉色背景干扰）。
3. 打开福袋面板：
   - 出现「一键发表评论」→ 点击完成福袋任务
   - 显示「任务已完成 等待开奖」→ 已参与，关闭面板继续
4. 读面板倒计时排开奖时间；临近开奖进 90 秒观察窗口，识别「中奖 / 未中奖」并记录。
5. 自适应节奏：剩余 >10 分钟每 3 分钟查一次，>4 分钟每 90 秒，临近开奖加密到 30 秒内。
6. 常亮保护：脚本外已设置 `screen_off_timeout` 最大、`stay_on_while_plugged_in`；截图失败会自动重连 adb、重启 adb server、退回 shell+pull。

## 目录与输出

```
fudai_auto.py        主脚本
bag_template.png     福袋图标模板
requirements.txt     依赖
log/                 运行输出（日志与截图统一放这里）
log/run_*.log        每次运行的文本日志
log/fudai_log.csv    参与/开关播/开奖结果记录
log/shots/           关键节点截图
.venv/               虚拟环境
```

## 注意

- 参与福袋可能要求发送直播间指定评论，脚本只点「一键发表评论」，不输入自定义文本。
- 必须停留在直播间：开奖前离开会失去资格，脚本不会点关闭/返回。
- 单实例锁：脚本占用本地 59999 端口，重复启动会直接退出。
