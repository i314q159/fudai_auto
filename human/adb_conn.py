import subprocess


def get_adb_device_serial_via_subprocess():
    """取 adb devices 里第一台「已授权（device）」的设备序列号。

    没有 adb、adb 执行失败、没有设备、设备 offline/unauthorized 时都返回 None，
    由调用方给出明确提示，而不是在 import 阶段抛异常。
    """
    try:
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"adb devices 执行失败: {e}")
        return None

    for line in result.stdout.splitlines()[1:]:
        parts = line.split("\t")
        # 只有 "serial\tdevice" 才算可用；offline / unauthorized 跳过
        if len(parts) >= 2 and parts[1].strip() == "device":
            return parts[0].strip()
    return None
