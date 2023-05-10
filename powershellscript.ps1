$commitMessage = "commit message"
for ($i = 1; $i -le 2; $i++) {
    "BATCH_NUM=$i" | Out-File -FilePath ".env" -Append
    git add .
    git commit -m $commitMessage
    git push
}
