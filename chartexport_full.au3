#cs ----------------------------------------------------------------------------

 AutoIt Version: 3.3.14.2
 Author:         daiwei

 All numbers in this script is set to co-operate with Observer8.3.5 fullscreen
 mode under 1920*1080

 Use the default English input method of Windows(os name).

 Close background programs at most to speedup and avoid pop-messages


#ce ----------------------------------------------------------------------------

; Script Start

#include <AutoItConstants.au3>

HotKeySet("{ESC}", "Terminate")	; register ESC as terminate button


MouseClick($MOUSE_CLICK_LEFT, 600, 1050)	; click Windows taskbar to focus on Observer
										    ; the number 600 should be adjust according to the place of Observer on taskbar



;
; External loop
;
; To click and open windturbines one by one
;
; Make sure that Observer is under "path tree view" and no windturbine is expanded
;

Local $turbine_num = 39
Local $signal_num = 44

For $t = 36 To $turbine_num - 1


   Local $y_place = 192 + 16 * $t

   MouseClick($MOUSE_CLICK_LEFT, 58, $y_place)	; click "+" to expand the 't'th windturbine

   sleep(200)

   If $t < 6 Then
	  MouseClick($MOUSE_CLICK_LEFT, 160, $y_place + 16)		; click 'generator speed' for #1-#6 windturbine
   ElseIf $t == 9 Or $t == 15 Then
	  MouseClick($MOUSE_CLICK_LEFT, 160, 256 + 16)			; click 'generator speed' for #10\#16 windturbine
   Else
	  MouseClick($MOUSE_CLICK_LEFT, 160, 272 + 16)			; click 'generator speed' for others windturbine
   EndIf



   ;
   ; Internal loop
   ;
   ; Note that before internal loop, cursor in the Observer window should be focused on "generator speed" of
   ; certain windturbine
   ;
   ; One epoch will export all signal tunnels' spectrum data of ONE windturbine on ONE day
   ;

   For $i = 0 To $signal_num - 1

      ; step 1, choose tunnel

      MouseClick($MOUSE_CLICK_LEFT, 140, 100)	; click "show next observation"

      MouseClick($MOUSE_CLICK_LEFT, 400, 60)	; click "spectrum"

      sleep(500)								; wait for GUI response


      ; step 2, focus on certain date


      MouseClick($MOUSE_CLICK_LEFT, 280, 100)	; click "date" button

      sleep(200)

      MouseClick($MOUSE_CLICK_LEFT, 360, 170)

      sleep(200)

      MouseClick($MOUSE_CLICK_LEFT, 360, 170)

      sleep(200)

      MouseClick($MOUSE_CLICK_LEFT, 300, 250)	; choose year 2013

      sleep(200)

      MouseClick($MOUSE_CLICK_LEFT, 300, 290)	; choose month 9

      sleep(200)

      MouseClick($MOUSE_CLICK_LEFT, 290, 297)	; choose date 30

      sleep(500)

	  ; 2013.12.31 is (300, 250), (435, 290), (320, 297)
	  ; 2013.11.30 is (300, 250), (400, 290), (430, 285)
	  ; 2013.10.31 is (300, 250), (350, 290), (380, 280)
	  ; 2013.09.30 is (300, 250), (300, 290), (290, 297)
	  ; 2013.09.27 is (300, 250), (300, 290), (412, 285)
	  ; 2013.08.31 is (300, 250), (435, 250), (430, 285)
	  ; 2013.07.31 is (300, 250), (400, 250), (350, 297)


      ; step 3, export and save jpg

      MouseClick($MOUSE_CLICK_RIGHT, 600, 400)	; draw right-click menu

      MouseClick($MOUSE_CLICK_LEFT, 650, 660)	; call export window

      MouseClick($MOUSE_CLICK_LEFT, 800, 485)   ; choose 'JPEG'

      MouseClick($MOUSE_CLICK_LEFT, 880, 630)   ; save button

      Sleep(100)

	  Send("spec.0930.t" & $t & ".s" & $i & ".jpg")		    ;t0 is #1 Windturbine, s0 is MB VEL (low class). t for windturbine, s for signal tunnel
															;too many keyboard input would extremely slow down the running speed
	  Sleep(100)

      Send("{Enter}")

      Sleep(200)


      ; step 4, export and save txt

      MouseClick($MOUSE_CLICK_LEFT, 880, 430)	; data button

      MouseClick($MOUSE_CLICK_LEFT, 880, 630)	; save button

      Sleep(100)

      Send("spec.0930.t" & $t & ".s" & $i & ".txt")

      Sleep(100)

      Send("{Enter}")

      Sleep(200)


      ; step 5, prepare for next loop

      MouseClick($MOUSE_CLICK_LEFT, 1130, 400)	 ; close export window

	  MouseClick($MOUSE_CLICK_LEFT, 1910, 30)	; close the charts' window opened


   Next


   sleep(500)

   ; click the "-" of the 't'th windturbine expanded
   If $t < 6 Then
	  MouseClick($MOUSE_CLICK_LEFT, 58, $y_place)
   ElseIf $t == 9 Or $t == 15 Then
	  MouseClick($MOUSE_CLICK_LEFT, 58, 256)
   Else
	  MouseClick($MOUSE_CLICK_LEFT, 58, 272)
   EndIf



Next


Func Terminate()
    Exit
EndFunc   ;==>Terminate