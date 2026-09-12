@echo off
echo Building AI Architect Textbook...
echo.

REM Check if LaTeX is installed
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found. Please install TeX Live or MiKTeX.
    echo Download TeX Live: https://www.tug.org/texlive/
    echo Download MiKTeX: https://miktex.org/download
    pause
    exit /b 1
)

where biber >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: biber not found. Please install TeX Live or MiKTeX.
    echo Download TeX Live: https://www.tug.org/texlive/
    echo Download MiKTeX: https://miktex.org/download
    pause
    exit /b 1
)

echo Step 1: Running pdflatex (first pass)...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: pdflatex failed on first pass.
    pause
    exit /b 1
)

echo Step 2: Running biber for bibliography...
biber main
if %errorlevel% neq 0 (
    echo ERROR: biber failed.
    pause
    exit /b 1
)

echo Step 3: Running pdflatex (second pass)...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: pdflatex failed on second pass.
    pause
    exit /b 1
)

echo Step 4: Running pdflatex (third pass)...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: pdflatex failed on third pass.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Output: main.pdf
echo.

REM Clean up auxiliary files
echo Cleaning up auxiliary files...
del /q *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.run.xml *.fdb_latexmk *.fls *.synctex.gz 2>nul

echo Done!
pause