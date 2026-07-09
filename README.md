
# IP Webcam Viewer

Small Python client for the Android **IP Webcam** app.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use

1. Connect your computer and phone to the same Wi-Fi network.
2. Open the **IP Webcam** app on your phone.
3. Tap **Start server**.
4. Copy the address shown on the phone, usually like `http://192.168.1.23:8080`.
5. Run:

```bash
python3 ip_webcam_viewer.py 192.168.1.23:8080
```

Press `s` to save a snapshot in `snapshots/`.
Press `q` or `Esc` to quit.

## Possible Weapon Detection

This app can draw alerts for objects that the model labels as possible weapons.
When a weapon label is detected, it will:

- draw a red box on the video
- print `ALERT: possible weapon detected` in the terminal
- save an alert image in `alerts/`

```bash
python3 ip_webcam_viewer.py 192.168.1.23:8080 --detect-weapons
```

For your trained weapon model, run:

```bash
python3 ip_webcam_viewer.py 192.168.150.47:8080 --detect-weapons
```

This opens a popup window named `Phone IP Webcam - Weapon Detection`. If the
phone camera sees a trained weapon class, the app draws a red rectangle and label
on the live video. The current trained classes are:

```text
Automatic Rifle, Bazooka, Grenade Launcher, Handgun, Knife, SMG, Shotgun, Sniper, Sword
```

You can adjust the confidence threshold:

```bash
python3 ip_webcam_viewer.py 192.168.1.23:8080 --detect-weapons --confidence 0.6
```

The default detection model path is now:

```text
runs/detect/train-3/weights/best.pt
```

Important: the default `yolov8n.pt` model usually does **not** include real gun
classes like `gun`, `pistol`, or `rifle`. For real firearm detection, use a
custom YOLO weapon model:

```bash
python3 ip_webcam_viewer.py 192.168.1.23:8080 --detect-weapons --model best.pt
```

If your custom model uses different class names, pass them like this:

```bash
python3 ip_webcam_viewer.py 192.168.1.23:8080 --detect-weapons --model best.pt --weapon-labels gun,pistol,rifle
```

This detection is only a helper. It can miss real weapons and can also create
false alerts.

## Weapon And Fight Detection

Use both trained models with the phone IP Webcam. Both weapon and fight
detection now start by default:

```bash
source .venv/bin/activate
python3 ip_webcam_viewer.py 192.168.150.47:8080
```

For lower delay, use the faster settings:

```bash
python3 ip_webcam_viewer.py 192.168.150.47:8080 --imgsz 320 --process-every 3 --stream-width 640 --stream-height 480
```

The popup shows `Normal` when no weapon or fight is detected. If something is
detected, it shows the detected label, draws a rectangle, prints an alert in the
terminal, and saves an alert image in `alerts/`.

Default model paths:

```text
Weapon: runs/detect/train-3/weights/best.pt
Fight:  runs/detect/train-5/weights/best.pt
```

You can also run only one detector:

```bash
python3 ip_webcam_viewer.py 192.168.150.47:8080 --detect-weapons
python3 ip_webcam_viewer.py 192.168.150.47:8080 --detect-fight
```

to run the app
```bash
python3 ip_webcam_viewer.py 192.168.150.47:8080 --detect-weapons
```

