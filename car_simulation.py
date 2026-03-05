import pygame
import cv2
import numpy as np
import tensorflow as tf
import os
import math

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "traffic_hand_signal_cnn.h5"
DATASET_PATH = "DataSet"
IMG_SIZE = (224, 224)

CONF_THRESHOLD = 0.0   # Accept ANY confidence
LANE_CHANGE_SPEED = 0.12   # Fast lane switching

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted(os.listdir(DATASET_PATH))
print("Classes:", class_names)

# ===============================
# PYGAME SETUP
# ===============================
pygame.init()
WIDTH, HEIGHT = 1100, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fast Response Autonomous Simulation")

clock = pygame.time.Clock()

# Colors
GRASS = (30, 140, 30)
ROAD = (50, 50, 50)
LANE = (230, 230, 230)
BORDER = (255, 255, 255)

CAR_BLUE = (30, 140, 255)
WINDOW = (180, 220, 255)
LIGHT = (255, 220, 100)
TIRE = (30, 30, 30)

# ===============================
# TRACK SETTINGS (3 LANES)
# ===============================
cx, cy = WIDTH // 2, HEIGHT // 2

OUTER = 340
LANE_W = 45

LANE_RIGHT = OUTER
LANE_MID = OUTER - LANE_W
LANE_LEFT = OUTER - 2 * LANE_W
INNER = OUTER - 3 * LANE_W

lane_radii = [LANE_LEFT, LANE_MID, LANE_RIGHT]

# ===============================
# CAR STATE
# ===============================
angle = 0
speed = 1
ANG_SPEED = 0.023

current_lane = 1   # Start middle
target_lane = 1

# Webcam
cap = cv2.VideoCapture(0)


# ===============================
# DRAW CAR
# ===============================
def draw_car(x, y, angle):

    car = pygame.Surface((80, 40), pygame.SRCALPHA)

    pygame.draw.rect(car, CAR_BLUE, (0, 12, 80, 22), border_radius=8)
    pygame.draw.rect(car, (20,110,210), (18, 0, 45, 16), border_radius=6)

    pygame.draw.rect(car, WINDOW, (22, 4, 15, 10))
    pygame.draw.rect(car, WINDOW, (43, 4, 15, 10))

    pygame.draw.circle(car, LIGHT, (74, 18), 4)
    pygame.draw.circle(car, LIGHT, (74, 25), 4)

    for xw in [15, 65]:
        pygame.draw.circle(car, TIRE, (xw, 8), 4)
        pygame.draw.circle(car, TIRE, (xw, 32), 4)

    shadow = pygame.Surface((90, 50), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0,0,0,60), (5,25,70,15))

    rot = pygame.transform.rotate(car, -math.degrees(angle))
    rect = rot.get_rect(center=(x, y))

    srot = pygame.transform.rotate(shadow, -math.degrees(angle))
    srect = srot.get_rect(center=(x+3, y+6))

    screen.blit(srot, srect)
    screen.blit(rot, rect)


# ===============================
# CONTROL LOGIC
# ===============================
def control_car(label, conf):

    global speed, target_lane

    if conf < CONF_THRESHOLD:
        return

    label = label.lower()

    # Stop ONLY on red
    if "red" in label or "stop" in label:
        speed = 0
        return

    # Otherwise always move
    speed = 1

    # Instant lane change
    if "left" in label and target_lane > 0:
        target_lane -= 1

    elif "right" in label and target_lane < 2:
        target_lane += 1


# ===============================
# DRAW ROAD
# ===============================
def draw_road():

    pygame.draw.circle(screen, ROAD, (cx, cy), OUTER)
    pygame.draw.circle(screen, GRASS, (cx, cy), INNER)

    pygame.draw.circle(screen, BORDER, (cx, cy), OUTER, 4)
    pygame.draw.circle(screen, BORDER, (cx, cy), INNER, 4)

    pygame.draw.circle(screen, LANE, (cx, cy), LANE_MID, 2)
    pygame.draw.circle(screen, LANE, (cx, cy), LANE_LEFT, 2)

    for a in range(0, 360, 15):

        r = math.radians(a)

        for rad in [LANE_LEFT, LANE_MID]:

            x1 = cx + (rad - 4) * math.cos(r)
            y1 = cy + (rad - 4) * math.sin(r)

            x2 = cx + (rad + 4) * math.cos(r)
            y2 = cy + (rad + 4) * math.sin(r)

            pygame.draw.line(screen, LANE, (x1, y1), (x2, y2), 2)


# ===============================
# MAIN LOOP
# ===============================
running = True

while running:

    clock.tick(60)
    screen.fill(GRASS)

    draw_road()

    # ===============================
    # CNN INPUT
    # ===============================
    ret, frame = cap.read()

    label = "No Signal"
    conf = 0

    if ret:

        frame = cv2.flip(frame, 1)

        img = cv2.resize(frame, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, 0)

        pred = model.predict(img, verbose=0)[0]

        cid = np.argmax(pred)
        conf = pred[cid]

        label = class_names[cid]

        control_car(label, conf)

        cv2.putText(
            frame,
            f"{label} {conf*100:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Signal Detection", frame)

    # ===============================
    # FAST LANE CHANGE
    # ===============================
    if current_lane < target_lane:
        current_lane += LANE_CHANGE_SPEED

    elif current_lane > target_lane:
        current_lane -= LANE_CHANGE_SPEED

    lane_index = round(current_lane)

    # ===============================
    # UPDATE POSITION
    # ===============================
    angle += speed * ANG_SPEED

    radius = lane_radii[lane_index]

    car_x = cx + radius * math.cos(angle)
    car_y = cy + radius * math.sin(angle)

    draw_car(car_x, car_y, angle)

    # ===============================
    # HUD
    # ===============================
    font = pygame.font.SysFont("arial", 26)

    hud = font.render(
        f"Lane: {lane_index+1} | Signal: {label}",
        True,
        (255,255,255)
    )

    screen.blit(hud, (20,20))

    pygame.display.update()

    # ===============================
    # EXIT
    # ===============================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False


# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
pygame.quit()
