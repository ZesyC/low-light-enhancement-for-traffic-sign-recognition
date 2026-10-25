param([string]$Compiler = 'xelatex')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force '_build' | Out-Null
    if ([System.IO.Path]::GetFileNameWithoutExtension($Compiler) -eq 'tectonic') {
        & $Compiler --keep-logs --keep-intermediates --outdir _build main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed' }
    }
    else {
        for ($pass = 1; $pass -le 3; $pass++) {
            & $Compiler --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
            if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed on pass $pass" }
        }
    }
    Copy-Item -LiteralPath '_build/main.pdf' -Destination '../bao_cao.pdf'
    Write-Output "PDF: $PSScriptRoot\..\bao_cao.pdf"
}
finally { Pop-Location }
