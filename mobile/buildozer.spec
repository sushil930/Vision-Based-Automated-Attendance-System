[app]
title = AI Attendance
package.name = aiattendance
package.domain = org.attendance
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0.0
requirements = python3,kivy==2.3.1,kivymd==2.0.0,requests,pillow,certifi,charset-normalizer,idna,urllib3,materialyoucolor,asynckivy,asyncgui,materialshapes
orientation = portrait
fullscreen = 0
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,CAMERA
android.api = 33
android.minapi = 21
android.arch = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
