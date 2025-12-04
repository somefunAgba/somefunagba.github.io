@echo off
setlocal

REM detect current branch
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" (
  echo Error: not a git repo or git not found.
  exit /b 1
)

echo Staging all changes...
git add .

echo Committing with message "update"...
git commit -m "update" || (
  echo Commit failed or no changes to commit.
  exit /b 0
)

echo Pushing to origin/%BRANCH%...
git push origin "%BRANCH%"

endlocal