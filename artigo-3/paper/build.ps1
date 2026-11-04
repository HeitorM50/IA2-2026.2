[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Invoke-LatexCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Comando '$Command' não encontrado. Instale o MiKTeX antes de compilar."
    }

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Command' terminou com código $LASTEXITCODE."
    }
}

Push-Location $PSScriptRoot
try {
    $latexArguments = @(
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex"
    )

    Invoke-LatexCommand "pdflatex" $latexArguments
    Invoke-LatexCommand "bibtex" @("main")
    Invoke-LatexCommand "pdflatex" $latexArguments
    Invoke-LatexCommand "pdflatex" $latexArguments

    $undefined = Select-String -Path "main.log" -Pattern @(
        "There were undefined references",
        "Citation .* undefined",
        "Reference .* undefined"
    )
    if ($undefined) {
        throw "A compilação terminou com citações ou referências indefinidas."
    }

    Write-Host "Artigo compilado com sucesso: $PSScriptRoot\main.pdf"
}
finally {
    Pop-Location
}
