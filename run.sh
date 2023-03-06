#!/bin/bash

/usr/bin/python3 /home/pi/flask_cam/app.py &
sleep 10
chromium-browser --kiosk http://0.0.0.0:8000 &
