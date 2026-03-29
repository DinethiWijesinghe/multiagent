param(
    [Parameter(Mandatory = $true)]
    [string]$Url
)

$envFile = Join-Path $PSScriptRoot "multiagent\.env"

if ($Url -notmatch '^https://[A-Za-z0-9-]+\.trycloudflare\.com/?$') {
    Write-Error "Provide a valid trycloudflare URL, for example: https://example-name.trycloudflare.com"
    exit 1
}

$normalizedUrl = $Url.TrimEnd('/')
$content = @(
    '# Active Colab tunnel backend URL'
    "VITE_API_URL=$normalizedUrl"
)

Set-Content -Path $envFile -Value $content

Write-Host "Updated $envFile"
Write-Host "VITE_API_URL=$normalizedUrl"