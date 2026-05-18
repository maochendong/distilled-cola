import subprocess, os, sys, time

os.chdir("/Users/Darkknight/Documents/蒸馏小可乐")
env = os.environ.copy()
env["TEACHER_MODEL"] = "deepseek-v4-flash"

with open("/tmp/trans_run.log", "w") as f:
    proc = subprocess.Popen(
        [".venv/bin/python", "-u", "scripts/ingest_transcriptions.py"],
        stdout=f, stderr=subprocess.STDOUT, env=env
    )
    print(f"PID: {proc.pid}")

    # Wait up to 30 minutes
    try:
        proc.wait(timeout=1800)
        print(f"Exit code: {proc.returncode}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Timed out after 30min", file=sys.stderr)
