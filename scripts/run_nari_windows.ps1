param(
  [string]$Distribution = "Debian",
  [switch]$Setup,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ServerArguments
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslRepository = (& wsl.exe -d $Distribution -- wslpath -a -u $repository).Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslRepository) {
  throw "Unable to resolve the Voice Open repository inside WSL distribution $Distribution."
}

if ($Setup) {
  & wsl.exe -d $Distribution --cd $wslRepository -- bash scripts/setup_nari_wsl.sh
} else {
  & wsl.exe -d $Distribution --cd $wslRepository -- env VOICE_QWEN_ENGINE=nari `
    bash scripts/run_nari_wsl.sh @ServerArguments
}
exit $LASTEXITCODE
