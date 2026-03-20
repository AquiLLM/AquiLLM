# Fail if generated or local dependency paths are tracked (run from repo root).
$ErrorActionPreference = "Stop"
$bannedPatterns = @(
    "^node_modules/",
    "^aquillm/tmp/"
)
$tracked = git ls-files 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "git ls-files failed (are you at the repository root?)"
    exit 1
}
$bad = @()
foreach ($path in $tracked) {
    foreach ($pat in $bannedPatterns) {
        if ($path -match $pat) {
            $bad += $path
        }
    }
}
if ($bad.Count -gt 0) {
    Write-Error ("Tracked paths match hygiene ban: " + ($bad -join ", "))
    exit 1
}
exit 0
