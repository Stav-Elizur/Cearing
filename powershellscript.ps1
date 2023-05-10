for ($i = 1; $i -le 2; $i++) {
    $line = "BATCH_SIZE=$i"
    (Get-Content -Path ".env") -replace "BATCH_SIZE=\d+", $line | Set-Content -Path ".env"
    git add .
    git commit -m "commit message $i"
    git push
}
