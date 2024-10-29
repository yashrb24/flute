import serial
import time
import csv
import cv2
import threading
import os

MySerial = serial.Serial(port = "/dev/ttyUSB0", baudrate=460800)
print("Stabilizing sensor...")
time.sleep(2)

# Video recording function to run in separate thread
def record_video(filename, duration):
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename + '.avi', fourcc, fps, (frame_width, frame_height))
    
    end_time = time.perf_counter() + duration
    
    while time.perf_counter() < end_time:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
    
    # Release everything when done
    cap.release()
    out.release()

ax = 0.0
ay = 0.0
az = 0.0
s = ""
i = 0
base_filename = input("Enter file name: ")
filename = os.path.join("recorded_data", base_filename + ".csv")
video_filename = os.path.join("recorded_data", base_filename + "_video")

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

# Start video recording in a separate thread
duration = 150  # 5 seconds of recording
video_thread = threading.Thread(target=record_video, args=(video_filename, duration))
video_thread.start()

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ax', 'ay', 'az'])

    print("Samples Written = ", end="")

    end_time = time.perf_counter() + duration  # Recording 5s of data

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

MySerial.close()
video_thread.join()  # Wait for video recording to complete
print("\nRecording completed!")
