# Emergency camera dashboard (web)

Files: `web_app.py` and `index.html` (in the same folder, or put `index.html` in a `static` folder).

```powershell
pip install -r requirements.txt
python web_app.py
```
Open http://127.0.0.1:5000

## What it detects
- **Fire / smoke**: built-in flickering-flame detector (no model file) and/or a custom model in `models/fire.pt`.
- **Person down and not moving**: `yolo11n-pose.pt` (downloads on first run) finds people; the app alerts
  when someone is lying down and motionless for the chosen number of seconds (default 15).
  If the pose model can't download, it falls back to `yolo11n.pt` (less accurate).

No restricted-zone feature. Alerts show in the dashboard, play a sound, save a snapshot in `alerts/`,
and can go to your phone through ntfy.

## Limits
A camera cannot tell if someone is unconscious. It sees "lying down and not moving", which also happens when
someone sleeps or rests on the floor. Always verify alerts. The built-in fire detector can be fooled by bright
orange flickering lights; it is not a replacement for a certified fire alarm.

## Phone notifications (ntfy)
1. Install the free **ntfy** app on your phone (Android / iPhone).
2. In the dashboard click **Generate** next to "Phone alert topic".
3. In the ntfy app tap **+** and subscribe to that exact topic name.
4. Click **Send test phone alert** in the dashboard. It tells you if it worked.
5. Start detection. Fire and person-down alerts arrive as urgent notifications with a snapshot photo.

The PC running `web_app.py` needs internet access to reach ntfy.sh. Anyone who knows your topic can read the alerts, so don't share it.
