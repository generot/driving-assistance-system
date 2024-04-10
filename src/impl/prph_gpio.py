from subprocess import check_call

from gpiozero import Button, LED, Buzzer
from signal import pause

B1_GPIO = 27
B2_GPIO = 8
B3_GPIO = 9
B4_GPIO = 10

LED_WARN_GPIO = 20
LED_ALERT_GPIO = 21

BUZZER_GPIO = 19

periphery_table = {
    "but_1": Button(B1_GPIO),
    "but_2": Button(B2_GPIO),
    "but_3": Button(B3_GPIO),
    "but_4": Button(B4_GPIO),
    "led_warn": LED(LED_WARN_GPIO),
    "led_alert": LED(LED_ALERT_GPIO),
    "buzzer": Buzzer(BUZZER_GPIO)
}

def shutdown_device():
    check_call([ "shutdown", "now" ])

periphery_table["but_4"].when_pressed = shutdown_device
