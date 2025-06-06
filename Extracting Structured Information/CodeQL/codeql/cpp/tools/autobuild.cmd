@echo off

if "%CODEQL_EXTRACTOR_CPP_BUILD_MODE%"=="none" (
python3 "%CODEQL_EXTRACTOR_CPP_ROOT%\tools\extract-standalone.py"
exit /b %ERRORLEVEL%
)

rem Uses the C# autobuilder to discover and trace msbuild projects

rem The autobuilder is already being traced
set CODEQL_AUTOBUILDER_CPP_NO_INDEXING=true

type NUL && "%CODEQL_EXTRACTOR_CPP_ROOT%/tools/%CODEQL_PLATFORM%/Semmle.Autobuild.Cpp.exe"
exit /b %ERRORLEVEL%
