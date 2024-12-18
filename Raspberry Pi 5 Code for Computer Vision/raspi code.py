import cv2
from picamera2 import Picamera2
import serial
from time import sleep

# Load Haar Cascade for upper body detection
haar_upper_body_cascade = cv2.CascadeClassifier("/home/saksham/Downloads/haarcascade_upperbody.xml")
cv2.startWindowThread()

# Initialize Picamera and configure preview
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Set up Bluetooth and Arduino serial communication
blut = serial.Serial("/dev/rfcomm0", 9600)
sleep(1)  # Wait for Bluetooth connection to establish
ArduinoSerial = serial.Serial("/dev/ttyACM0", 9600)
sleep(2)  # Wait for Arduino connection to establish

# Tracking parameters
x_deviation, y_max, tolerance = 0, 0, 0.05
frame_center_x, frame_center_y = 640 // 2, 480 // 2
green_line_offset = 100  # Increased for less sensitivity

def track_object(upper_body):
    global x_deviation, y_max
    
    if len(upper_body) == 0:
        print("No objects to track")
        blut.write(b'X0S')  # Stop if no object is detected
        return

    # Track the first detected upper body
    x, y, w, h = upper_body[0]
    x_min, y_min, x_max, y_max = x, y, x + w, y + h

    # Calculate Deviation and center
    obj_x_center = x_min + w // 2
    obj_y_center = y_min + h // 2
    obj_x_center_norm = round(obj_x_center / 640, 3)  # Normalize to frame width
    x_deviation = round(0.5 - obj_x_center_norm, 3)    # Deviation from center
    y_max = round(y_max / 480, 3)                      # Normalize to frame height

    # Send x, y coordinates to Arduino if outside threshold box
    if abs(obj_x_center - frame_center_x) > green_line_offset or abs(obj_y_center - frame_center_y) > green_line_offset:
        coord_string = 'X{0:d}Y{1:d}'.format(obj_x_center, obj_y_center)
        ArduinoSerial.write(coord_string.encode('utf-8'))
    
    print("{", x_deviation, y_max, "}")
    move_robot()  # Call movement function based on deviation

def move_robot():
    global x_deviation, y_max, tolerance
    
    y_distance = 1 - y_max  # Distance from bottom of frame
    if abs(x_deviation) < tolerance:
        if y_distance < 0.1:
            blut.write(b'S\n')
            print("Reached person...........")
        else:
            blut.write(b'F\n')
            print("Moving robot ...FORWARD....")
    else:
        if x_deviation > tolerance:
            blut.write(b'X0D')  # Command to move left
            print("Moving robot ...Left....")
        elif x_deviation < -tolerance:
            blut.write(b'X0O')  # Command to move right
            print("Moving robot ...Right....")

# Main loop
while True:
    im = picam2.capture_array()
    im = cv2.flip(im, flipCode=1)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Perform tracking and movement control
    track_object(upper_body)

    # Draw bounding boxes and center markers
    for x, y, w, h in upper_body:
        cv2.circle(im, (x + w // 2, y + h // 2), 2, (0, 255, 0), 2)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Calculate and display distance
        focal_length = 4.75 * 100
        roi_gray = gray[y:y+h, x:x+w]
        length, breadth = roi_gray.shape[0], roi_gray.shape[1]
        distance = (length * breadth) / focal_length
        cv2.putText(im, 'Distance =' + str(distance), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the central boundary box for adjustments (modified size to match green lines)
    cv2.rectangle(im, (frame_center_x - green_line_offset, frame_center_y - 40),
                  (frame_center_x + green_line_offset, frame_center_y + 40),
                  (255, 255, 255), 3)
    cv2.line(im, (frame_center_x - green_line_offset, 0), 
             (frame_center_x - green_line_offset, 480), (0, 255, 0), 2)
    cv2.line(im, (frame_center_x + green_line_offset, 0), 
             (frame_center_x + green_line_offset, 480), (0, 255, 0), 2)

    cv2.imshow('Frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
