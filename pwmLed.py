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
    
def ledBrightness(duty):
    duty = int(duty*255/100)
    print(duty)
    pi.set_PWM_dutycycle(18, duty)
    return