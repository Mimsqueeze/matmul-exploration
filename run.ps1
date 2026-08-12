param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name
)

$srcFile = Join-Path $PSScriptRoot "src\$Name.cu"
$outputDir = Join-Path $PSScriptRoot "output"
$outFile = Join-Path $outputDir "$Name.exe"
$ncuLog = Join-Path $outputDir "$Name.ncu.log"

if (-not (Test-Path $srcFile)) {
    Write-Error "File not found: $srcFile"
    exit 1
}

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vcvarsall = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $vcvarsall) {
        Write-Error "cl.exe not found and vcvarsall.bat could not be located. Install VS Build Tools with the C++ workload."
        exit 1
    }

    cmd /c "`"$vcvarsall`" x64 & set" | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
        }
    }
}

nvcc -arch=sm_86 -lcublas $srcFile -o $outFile
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $outFile
$exitCode = $LASTEXITCODE

# profile with Nsight Compute, if available -- run after the normal timed
# execution above so ncu's instrumentation overhead doesn't affect that number
if (Get-Command ncu -ErrorAction SilentlyContinue) {
    Write-Host "`n--- ncu profile (log: $ncuLog) ---"
    ncu --set basic -f --log-file $ncuLog $outFile
} else {
    Write-Warning "ncu not found on PATH -- skipping profiling. Install Nsight Compute (bundled with the CUDA toolkit) to enable it."
}

exit $exitCode
