Option Explicit
Dim action, shell, result, taskName
If WScript.Arguments.Count <> 1 Then WScript.Quit 2
action = LCase(WScript.Arguments(0))
If action <> "on" And action <> "off" Then WScript.Quit 2
taskName = "C500 Guard " & UCase(action)
Set shell = CreateObject("WScript.Shell")
result = shell.Run("""" & shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\schtasks.exe"" /Run /TN """ & taskName & """", 0, True)
If result <> 0 Then
    MsgBox "Windows could not switch BarrierWatcher " & UCase(action) & ". The power-control task needs repair. No successful change has been confirmed.", vbExclamation, "BarrierWatcher"
End If
WScript.Quit result
