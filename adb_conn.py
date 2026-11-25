import subprocess


def get_adb_device_serial_via_subprocess():
    try:
        # Run the command and capture output, checking for errors
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            # Split by tab to isolate the serial number
            device_serial = lines[1].split("\t")[0]
            return device_serial
    except subprocess.CalledProcessError as e:
        print(f"Error running adb: {e}")
    return None


device_id = get_adb_device_serial_via_subprocess()
