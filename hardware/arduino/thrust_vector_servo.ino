#include <Servo.h>

Servo servoX;
Servo servoY;
Servo servoZ;

const int pinX = 3;
const int pinY = 5;
const int pinZ = 6;

float thrustX = 0.0;
float thrustY = 0.0;
float thrustZ = 0.0;
int durationMs = 0;

int mapVectorToAngle(float v) {
  v = constrain(v, -1.0, 1.0);
  return (int)(90 + v * 60); // 90=center, +/-60 deg steering
}

void applyThrust(float x, float y, float z, int duration) {
  servoX.write(mapVectorToAngle(x));
  servoY.write(mapVectorToAngle(y));
  servoZ.write(mapVectorToAngle(z));

  delay(duration);

  servoX.write(90);
  servoY.write(90);
  servoZ.write(90);
}

void setup() {
  Serial.begin(115200);
  servoX.attach(pinX);
  servoY.attach(pinY);
  servoZ.attach(pinZ);

  servoX.write(90);
  servoY.write(90);
  servoZ.write(90);

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');

    // Expected CSV fallback: x,y,z,duration
    int p1 = msg.indexOf(',');
    int p2 = msg.indexOf(',', p1 + 1);
    int p3 = msg.indexOf(',', p2 + 1);

    if (p1 > 0 && p2 > p1 && p3 > p2) {
      thrustX = msg.substring(0, p1).toFloat();
      thrustY = msg.substring(p1 + 1, p2).toFloat();
      thrustZ = msg.substring(p2 + 1, p3).toFloat();
      durationMs = msg.substring(p3 + 1).toInt();

      applyThrust(thrustX, thrustY, thrustZ, durationMs);
      Serial.println("EXECUTED");
    } else {
      Serial.println("INVALID_COMMAND");
    }
  }
}
