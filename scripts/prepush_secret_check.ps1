param(
    [string]$RepoPath = "D:\Multiagent"
)

$ErrorActionPreference = "Stop"

Write-Host "Running pre-push secret checks in $RepoPath ..." -ForegroundColor Cyan

if (-not (Test-Path $RepoPath)) {
    Write-Host "FAIL: Repository path not found: $RepoPath" -ForegroundColor Red
    exit 2
}

$hits = @()

# 1) Real-looking Google API keys in tracked files
$googleKeyHits = git -C $RepoPath grep -nE "AIza[0-9A-Za-z_-]{35}" -- . 2>$null
if ($googleKeyHits) {
    $hits += "Found real-looking Google API key pattern in tracked files:"
    $hits += $googleKeyHits
}

# 2) GOOGLE_API_KEY assignment in tracked files (placeholder in .env.example is allowed)
$apiKeyVarHits = git -C $RepoPath grep -n "GOOGLE_API_KEY=" -- . 2>$null
$unsafeApiKeyVarHits = @()
if ($apiKeyVarHits) {
    foreach ($line in $apiKeyVarHits) {
        if ($line -notmatch "\.env\.example:" -and $line -notmatch "scripts/prepush_secret_check\.ps1:") {
            $unsafeApiKeyVarHits += $line
        }
    }
}
if ($unsafeApiKeyVarHits.Count -gt 0) {
    $hits += "Found GOOGLE_API_KEY assignment outside .env.example:"
    $hits += $unsafeApiKeyVarHits
}

# 3) Sensitive strings in staged added lines only
$stagedAddedLines = git -C $RepoPath diff --cached --no-color --unified=0 | Select-String -Pattern '^\+[^+]'
$stagedSensitiveHits = $stagedAddedLines | Select-String -Pattern 'AIza[0-9A-Za-z_-]{35}|GOOGLE_API_KEY\s*=\s*["'']?(?!your_key_here|your_real_key_here)[^"''\s]+|DATABASE_URL\s*=\s*(postgres|postgresql)://|npg_[0-9A-Za-z]{8,}'
if ($stagedSensitiveHits) {
    $hits += "Found sensitive patterns in staged changes:"
    $hits += ($stagedSensitiveHits | ForEach-Object { $_.Line })
}

if ($hits.Count -gt 0) {
    Write-Host "`nFAIL: Potential secrets detected." -ForegroundColor Red
    $hits | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host "`nAction: remove secrets from tracked/staged files and rotate if exposed." -ForegroundColor Yellow
    exit 1
}

Write-Host "PASS: No real secrets detected in tracked files or staged diff." -ForegroundColor Green
exit 0
