Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)
command = """" & projectFolder & "\.venv\Scripts\pythonw.exe"" """ & projectFolder & "\run.py"""
shell.Run command, 0, False

