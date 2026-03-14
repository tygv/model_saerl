param(
    [string]$PythonExe = "python",
    [string]$Tag = "source_context_v1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir ("saerl_phase3h_{0}_{1}.log" -f $Tag, $Timestamp)

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$env:PYTHONUNBUFFERED = "1"

$Cmd = @(
    "scripts/run_saerl_phase3h_source_context_v1.py",
    "--tag",
    $Tag
) + $ExtraArgs

Push-Location $RepoRoot
try {
    Write-Host ("[start] {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("[repo]  {0}" -f $RepoRoot)
    Write-Host ("[log]   {0}" -f $LogPath)
    Write-Host ("[run]   {0} {1}" -f $PythonExe, ($Cmd -join " "))

    & $PythonExe @Cmd 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        throw "Source-context run failed with exit code $ExitCode. See $LogPath"
    }

    Write-Host ("[done]  {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("[log]   {0}" -f $LogPath)
}
finally {
    Pop-Location
}
