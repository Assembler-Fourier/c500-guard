Option Explicit
' BarrierWatcher app launcher (no console window).
'   barrierwatcher.vbs            open the dashboard, starting BarrierWatcher first if it is off
'   barrierwatcher.vbs settings   the same, opening Settings
'   barrierwatcher.vbs off        switch the whole program off (same as Full OFF / F4)
'   barrierwatcher.vbs status     print "running" or "off" (for checks; run with cscript)
' It only starts the existing "C500 Guard ON"/"C500 Guard OFF" tasks; it changes no settings.

Const BASE = "http://127.0.0.1:8787"
Dim shell, files, mode, page, schtasks, result, waited
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
mode = "open"
If WScript.Arguments.Count > 0 Then mode = LCase(WScript.Arguments(0))
schtasks = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\schtasks.exe"

If mode = "status" Then
    If IsUp() Then WScript.Echo "running" Else WScript.Echo "off"
    WScript.Quit 0
End If

If mode = "off" Then
    result = shell.Run("""" & schtasks & """ /Run /TN ""C500 Guard OFF""", 0, True)
    If result <> 0 Then
        MsgBox "Windows could not start the BarrierWatcher shutdown. Use Full OFF on the dashboard, or press F4.", vbExclamation, "BarrierWatcher"
    Else
        shell.Popup "BarrierWatcher is switching off. Everything stops within a few seconds.", 3, "BarrierWatcher", vbInformation
    End If
    WScript.Quit result
End If

page = "/"
If mode = "settings" Then page = "/settings"

If Not IsUp() Then
    result = shell.Run("""" & schtasks & """ /Run /TN ""C500 Guard ON""", 0, True)
    If result <> 0 Then
        MsgBox "Windows could not start BarrierWatcher: its power task needs repair.", vbExclamation, "BarrierWatcher"
        WScript.Quit result
    End If
    shell.Popup "Starting BarrierWatcher. The dashboard opens when it is ready (usually 20-40 seconds).", 4, "BarrierWatcher", vbInformation
    waited = 4
    ' Wait while the ON task is still checking camera, AI and helpers (at most 2 minutes).
    Do While waited < 120 And Not IsArmed() And (waited < 8 Or TaskRunning("C500 Guard ON"))
        WScript.Sleep 1000
        waited = waited + 1
    Loop
    If Not IsUp() Then
        MsgBox "BarrierWatcher did not start." & vbCrLf & vbCrLf & _
               "Check that the capture stick is plugged in and the keyboard shows the ANPR Entrance camera, then open BarrierWatcher again.", _
               vbExclamation, "BarrierWatcher"
        WScript.Quit 1
    End If
End If
OpenPage BASE & page
WScript.Quit 0

Function Fetch(path)
    Dim http
    Fetch = ""
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    http.setTimeouts 1000, 1000, 2000, 2000
    http.open "GET", BASE & path, False
    http.send
    If Err.Number = 0 Then
        If http.status = 200 Then Fetch = http.responseText
    End If
    Err.Clear
    On Error GoTo 0
End Function

Function IsUp()
    IsUp = (Len(Fetch("/state")) > 0)
End Function

Function IsArmed()
    IsArmed = (InStr(Fetch("/state"), """armed"": true") > 0)
End Function

Function TaskRunning(name)
    Dim service, task
    TaskRunning = False
    On Error Resume Next
    Set service = CreateObject("Schedule.Service")
    service.Connect
    Set task = service.GetFolder("\").GetTask(name)
    If Err.Number = 0 Then TaskRunning = (task.State = 4)
    Err.Clear
    On Error GoTo 0
End Function

Sub OpenPage(url)
    Dim chrome
    chrome = shell.ExpandEnvironmentStrings("%ProgramFiles%") & "\Google\Chrome\Application\chrome.exe"
    If files.FileExists(chrome) Then
        ' An app window: no tabs or address bar, just BarrierWatcher.
        shell.Run """" & chrome & """ --app=" & url & " --window-size=1600,1000", 1, False
    Else
        shell.Run url, 1, False
    End If
End Sub
