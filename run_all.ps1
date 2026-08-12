# runs every .cu file in src/ through run.ps1 (build + run + ncu profile)
$srcDir = Join-Path $PSScriptRoot "src"
$runScript = Join-Path $PSScriptRoot "run.ps1"

$files = Get-ChildItem -Path $srcDir -Filter "*.cu" | Sort-Object Name
$results = @()

foreach ($file in $files) {
    $name = $file.BaseName
    Write-Host "`n===== $name =====" -ForegroundColor Cyan

    & $runScript $name
    $exitCode = $LASTEXITCODE

    $results += [PSCustomObject]@{
        Name     = $name
        ExitCode = $exitCode
        Success  = ($exitCode -eq 0)
    }
}

Write-Host "`n===== Summary =====" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failures = $results | Where-Object { -not $_.Success }
if ($failures.Count -gt 0) {
    Write-Warning "$($failures.Count) file(s) failed: $($failures.Name -join ', ')"
    exit 1
}

exit 0
