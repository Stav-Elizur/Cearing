for ($i = 1; $i -le 607; $i++) {
    "BATCH_NUM=$i" | Out-File -FilePath ".env" -Encoding utf8
    $content = Get-Content ".env"
    $content | Foreach-Object { $_ -replace '\s+', '' } | Set-Content ".env"
    git add .
    git commit -m "Generate signsuisse - separate $i"
    git push
}
