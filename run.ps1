param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name
)

$srcFile = Join-Path $PSScriptRoot "src\$Name.cu"
$outFile = Join-Path $PSScriptRoot "src\$Name.exe"
$expFile = Join-Path $PSScriptRoot "src\$Name.exp"
$libFile = Join-Path $PSScriptRoot "src\$Name.lib"

if (-not (Test-Path $srcFile)) {
    Write-Error "File not found: $srcFile"
    exit 1
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

nvcc -arch=sm_86 $srcFile -o $outFile
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $outFile
$exitCode = $LASTEXITCODE

# clean up scrap build files
Remove-Item -Path $outFile, $expFile, $libFile -ErrorAction SilentlyContinue

exit $exitCode
