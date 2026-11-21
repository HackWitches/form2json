# Podgląd postępu partii form2json na żywo (Ctrl-C aby zakończyć)
param([string]$Out = "C:\Hackathon2026\Formularze-Wypakowane\wyniki", [string]$Log = "C:\Hackathon2026\run_all.log", [int]$Total = 69)
while ($true) {
    Clear-Host
    $n = (Get-ChildItem -Path $Out -Filter *.form.json -ErrorAction SilentlyContinue).Count
    $last = Get-ChildItem -Path $Out -Filter *.form.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    Write-Host ("[{0}]  gotowe: {1}/{2}   GPU: {3}" -f (Get-Date -Format HH:mm:ss), $n, $Total, $gpu) -ForegroundColor Green
    if ($last) { Write-Host ("ostatni: {0} ({1:HH:mm:ss})" -f $last.Name, $last.LastWriteTime) }
    Write-Host "--- log VLM ---"
    Get-Content -Path $Log -Tail 12 -Encoding utf8
    if (-not (Get-Process -Name python -ErrorAction SilentlyContinue)) { Write-Host "Proces zakończony." -ForegroundColor Yellow; break }
    Start-Sleep -Seconds 3
}
