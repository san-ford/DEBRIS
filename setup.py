import subprocess
import sys

if sys.platform == "darwin" or sys.platform == "linux":
    cmd = "python3"
elif sys.platform == "win32":
    cmd = "py"
else:
    raise ValueError("Operating system not compatible")

subprocess.run(["cd", "debris"])
subprocess.run([cmd, "manage.py", "makemigrations"])
subprocess.run([cmd, "manage.py", "migrate"])
subprocess.run([cmd, "populate.py"])
subprocess.run([cmd, "manage.py", "runserver"])
