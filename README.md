windows限定
1. 該当folderを右クリック→「ターミナルで開く」（Windows PowerShell起動）

2. Windows PowerShellでの入力
Windows PowerShell起動直後の入力
```sh
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
```sh
.\setup_and_run.ps1 -OpenAIKey "YOUR API KEY" -Port 8000
```

reload
```sh
python launch.py
```