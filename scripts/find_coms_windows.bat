@ECHO OFF
SETLOCAL ENABLEEXTENSIONS 
SET Path=%Path%;C:\Users\Public\Documents\PySeq2500\Version1\
SET logpath=C:\Users\Public\Documents\PySeq2500\log\
SET me=%~n0
SET parent=%~dp0

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%ARM9CHEMA%%"" get Caption ^| findstr COM') do @set ARM9CHEM=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%ARM9DIAGA%%"" get Caption ^| findstr COM') do @set ARM9DIAG=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000004A%%"" get Caption ^| findstr COM') do @set FPGAcommand=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000005A%%"" get Caption ^| findstr COM') do @set FPGAresponse=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%KLOEHNAA%%"" get Caption ^| findstr COM') do @set KLOEHNA=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%KLOEHNBA%%"" get Caption ^| findstr COM') do @set KLOEHNB=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000006A%%"" get Caption ^| findstr COM') do @set LASER1=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000007A%%"" get Caption ^| findstr COM') do @set LASER2=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000008A%%"" get Caption ^| findstr COM') do @set PCcamera=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%PCIOA%%"" get Caption ^| findstr COM') do @set PCIO=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000010A%%"" get Caption ^| findstr COM') do @set TESTport=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%VICIA1A%%"" get Caption ^| findstr COM') do @set VICIA1=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%VICIA2A%%"" get Caption ^| findstr COM') do @set VICIA2=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%VICIB1A%%"" get Caption ^| findstr COM') do @set VICIB1=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%VICIB2A%%"" get Caption ^| findstr COM') do @set VICIB2=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000001A%%"" get Caption ^| findstr COM') do @set XSTAGE=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000003A%%"" get Caption ^| findstr COM') do @set ZSTAGE=%%d

for /f "tokens=1,2,3,4 delims= " %%a in ('wmic path CIM_LogicalDevice where "DeviceID like "%%IL000002A%%"" get Caption ^| findstr COM') do @set YSTAGE=%%d

echo $FPGAcommand
echo $FPGAresponse

# interactive_PySeq2500.py
