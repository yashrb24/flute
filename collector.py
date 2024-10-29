import serial
import time
import csv
import os

MySerial = serial.Serial(port = "/dev/ttyUSB0", baudrate=460800)
print("Stabilizing sensor...")
time.sleep(2)

ax = 0.0
ay = 0.0
az = 0.0
s = ""
i = 0
basename = input("Enter file name: ") + ".csv"
filename = os.path.join("recorded_data", basename)
def append(num, axis):
    global ax, ay, az
    if axis == 0:
        ax = num
    elif axis == 1:
        ay = num
    else:
        az = num
    return
    
count = 0

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ax', 'ay', 'az'])

    print("Samples Written = ", end="")

    end_time = time.perf_counter() + 5  # Recording 30s of data

    while time.perf_counter() < end_time:
        if MySerial.in_waiting > 0:       # Check if data is available
            print("\b" * len(str(count)), end="")
            byte = MySerial.read(size=1)  # Read 1 byte
            c = byte.decode('ascii')      # Decode byte to ASCII
            if c == ',':
                if s == '':
                    continue
                append(int(s),i)
                i += 1
            elif c == '\n':
                if s == '':
                    continue
                append(int(s),2)
                i = 0
                writer.writerow([ax, ay, az])

            count += 1
            print(f"{str(count)}", end="")

            if c.isdigit():
                s = s + c
            else:
                s = ""

