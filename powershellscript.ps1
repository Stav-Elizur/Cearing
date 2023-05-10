for ($i = 1; $i -le 2; $i++) {
    "BATCH_NUM=$i" | Out-File -FilePath ".env" -Encoding utf8
    git add .
    git commit -m "commit message $i"
    git push
}
