$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force '_build' | Out-Null
    for ($pass = 1; $pass -le 2; $pass++) {
        & xelatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed on pass $pass" }
    }
    Copy-Item -LiteralPath '_build/main.pdf' -Destination 'thuyet_trinh.pdf'
    Write-Output "PDF: $PSScriptRoot\thuyet_trinh.pdf"
}
finally {
    Pop-Location
}
