for ($i = 1; $i -le 60; $i++) {
    "BATCH_NUM=$i" | Out-File -FilePath ".env" -Encoding utf8
    $content = Get-Content ".env"
    $content | Foreach-Object { $_ -replace '\s+', '' } | Set-Content ".env"
    git add .
    git commit -m "commit message $i"
    git push
}
