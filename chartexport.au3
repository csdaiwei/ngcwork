#cs ----------------------------------------------------------------------------

 AutoIt Version: 3.3.14.2
 Author:         daiwei

 All numbers in this script is set to co-operate with Observer8.3.5 fullscreen
 mode under 1920*1080

 Use the default English input method of Windows(os name).


#ce ----------------------------------------------------------------------------

; Script Start

#include <AutoItConstants.au3>

HotKeySet("{ESC}", "Terminate")	; register ESC as terminate button


MouseClick($MOUSE_CLICK_LEFT, 600, 1050)	; click Windows taskbar to focus on Observer
										    ; the number 600 should be adjust according to the place of Observer on taskbar


Local $name[44]
$name[0] = "MB VEL (low class)"
$name[1] = "MB VEL (high class)"
$name[2] = "MB ACC (50X)"
$name[3] = "MB ENV1 (low class)"
$name[4] = "MB ENV1 (high class)"
$name[5] = "MB ENV3"
$name[6] = "Gear Input ACC (5kHz)"
$name[7] = "Gear Input ACC (low class)"
$name[8] = "Gear Input ACC (high class)"
$name[9] = "Gear Input ENV1 (low class)"
$name[10] = "Gear Input ENV1 (high class)"
$name[11] = "Gear Input ENV3"
$name[12] = "PI. stage ACC (5kHz)"
$name[13] = "PI. stage ACC (low class)"
$name[14] = "PI. stage ACC (high class)"
$name[15] = "PI. stage ENV1 (low class)"
$name[16] = "PI. stage ENV1 (high class)"
$name[17] = "PI. stage ENV3"
$name[18] = "LSS ACC (5kHz)"
$name[19] = "LSS ACC (low class)"
$name[20] = "LSS ACC (high class)"
$name[21] = "LSS ENV1 (low class)"
$name[22] = "LSS ENV1 (high class)"
$name[23] = "LSS ENV2"
$name[24] = "LSS ENV3"
$name[25] = "IMS ACC (low class)"
$name[26] = "IMS ACC (high class)"
$name[27] = "IMS ENV2"
$name[28] = "IMS ENV3 (low class)"
$name[29] = "IMS ENV3 (high class)"
$name[30] = "HSS ACC (5kHz)"
$name[31] = "HSS ACC (low class)"
$name[32] = "HSS ACC (high class)"
$name[33] = "HSS ENV3 (low class)"
$name[34] = "HSS ENV3 (high class)"
$name[35] = "Geno DS VEL (low class)"
$name[36] = "Geno DS VEL (high class)"
$name[37] = "Geno DS ENV3 (low class)"
$name[38] = "Geno DS ENV3 (high class)"
$name[39] = "Geno NDS ACC (5kHz)"
$name[40] = "Geno NDS VEL (low class)"
$name[41] = "Geno NDS VEL (high class)"
$name[42] = "Geno NDS ENV3 (low class)"
$name[43] = "Geno NDS ENV3 (high class)"

;
; main loop
;
; Note that before main loop, cursor in the Observer window should be focused on "generator speed" of
; certain windturbine under  "path tree view"
;
; One epoch will export all singal tunnels' spectrum data of ONE windturbine on ONE day
;


For $i = 0 To UBound($name)-1


   ; step 1, choose tunnel

   MouseClick($MOUSE_CLICK_LEFT, 140, 100)	; click "show next observation"

   MouseClick($MOUSE_CLICK_LEFT, 400, 60)	; click "spectrum"

   sleep(500)								; wait for GUI response


   ; step 2, focus on certain date (2013-10-31)

   MouseClick($MOUSE_CLICK_LEFT, 280, 100)	; click "date" button

   sleep(200)

   MouseClick($MOUSE_CLICK_LEFT, 360, 170)

   sleep(200)

   MouseClick($MOUSE_CLICK_LEFT, 360, 170)

   sleep(200)

   MouseClick($MOUSE_CLICK_LEFT, 300, 250)	; choose year 2013

   sleep(200)

   MouseClick($MOUSE_CLICK_LEFT, 350, 290)	; choose month 10

   sleep(200)

   MouseClick($MOUSE_CLICK_LEFT, 380, 280, 2)	; choose date 31

   sleep(500)


   ; step 3, export and save jpg

   MouseClick($MOUSE_CLICK_RIGHT, 600, 400)	; draw right-click menu

   MouseClick($MOUSE_CLICK_LEFT, 650, 660)	; call export window

   MouseClick($MOUSE_CLICK_LEFT, 800, 485)   ; choose 'JPEG'

   MouseClick($MOUSE_CLICK_LEFT, 880, 630)   ; save button

   Sleep(100)

   Send("spectrum " & $name[$i] & ".jpg")

   Sleep(100)

   Send("{Enter}")

   Sleep(500)


   ; step 4, export and save txt

   MouseClick($MOUSE_CLICK_LEFT, 880, 430)	; data button

   MouseClick($MOUSE_CLICK_LEFT, 880, 630)	; save button

   Sleep(100)

   Send("spectrum " & $name[$i] & ".txt")

   Sleep(100)

   Send("{Enter}")

   Sleep(500)


   ; step 5, prepare for next loop

   MouseClick($MOUSE_CLICK_LEFT, 1130, 400)	 ; close export window

   Sleep(100)

Next


Func Terminate()
    Exit
EndFunc   ;==>Terminate