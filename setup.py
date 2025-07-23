import subprocess
import sys

if sys.platform == "darwin" or sys.platform == "linux":
    cmd = "python3"
elif sys.platform == "win32":
    cmd = "py"
else:
    raise ValueError("Operating system not compatible")

subprocess.run([cmd, "debris/manage.py", "makemigrations"])
subprocess.run([cmd, "debris/manage.py", "migrate"])
subprocess.run([cmd, "debris/populate.py"])
