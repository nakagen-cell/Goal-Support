import subprocess
import webbrowser
import time
import sys
import os

URL = "http://127.0.0.1:8000/ui"

def main():
    print("LOAD: Starting uvicorn server...")

    # venv の Python をそのまま使って uvicorn を起動する
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--reload",
    ]
    print("CMD:", " ".join(cmd))

    proc = subprocess.Popen(cmd)

    # 少し待ってからブラウザを開く
    time.sleep(2)
    webbrowser.open(URL)
    print(f"OPENED: {URL}")

    try:
        # メインスレッドを uvicorn に張り付ける
        proc.wait()
    except KeyboardInterrupt:
        print("\nLOAD: KeyboardInterrupt 受信、サーバーを終了します...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("LOAD: terminate に失敗したため kill します。")
            proc.kill()

    print("DONE: サーバープロセスを終了しました。")

if __name__ == "__main__":
    main()
