import pigpio


pi = pigpio.pi()

# while True:
#     i = 0
#     while(i < 100):
#         pwm.ChangeDutyCycle(i)
#         i += 1
#         sleep(0.01)
#     while(i >= 0):
#         pwm.ChangeDutyCycle(i)
#         i -= 1
#         sleep(0.01)

def ledBrightness(pin, duty):
    duty = int(duty*255/100)
    print(duty)
    pi.set_PWM_dutycycle(pin, duty)
    return
