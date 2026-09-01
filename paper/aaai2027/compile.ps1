# Compile the AAAI 2027 submission (main paper + technical appendix).
# Requires a LaTeX distribution (MiKTeX or TeX Live) with pdflatex + bibtex.
#   MiKTeX:  winget install MiKTeX.MiKTeX   (then restart the shell)
# Run from this directory:  ./compile.ps1
$ErrorActionPreference = "Stop"

function Build-Doc($name) {
    pdflatex -interaction=nonstopmode "$name.tex"
    if (Test-Path "$name.aux") {
        $aux = Get-Content "$name.aux" -Raw
        if ($aux -match "\\citation") { bibtex $name }
    }
    pdflatex -interaction=nonstopmode "$name.tex"
    pdflatex -interaction=nonstopmode "$name.tex"
    Write-Output "Built $name.pdf"
}

Build-Doc "main"
Build-Doc "supp"

# Quick post-build sanity: unresolved references/citations
foreach ($log in @("main.log", "supp.log")) {
    if (Test-Path $log) {
        $warnings = Select-String -Path $log -Pattern "Warning.*(undefined|multiply)" -AllMatches
        if ($warnings) { Write-Output "CHECK ${log}:"; $warnings | ForEach-Object { $_.Line } }
    }
}
