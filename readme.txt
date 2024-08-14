CMD commands to run dev server. ONLY include execution policy line if on windows system.

Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate

DEACTIVATE


$env:FLASK_APP = "speedsight"
$env:FLASK_DEBUG = "1"    