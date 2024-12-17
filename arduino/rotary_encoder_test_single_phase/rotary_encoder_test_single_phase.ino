volatile int rotary_index = 0;
volatile bool clockwise = false;

int rotary_white_pin = 2;
bool white_value = LOW;

int rotary_green_pin = 3;
int green_value = LOW;

void setup() {
  Serial.begin(9600);

  pinMode(rotary_white_pin, INPUT_PULLUP);
  white_value = digitalRead(rotary_white_pin);
  attachInterrupt(digitalPinToInterrupt(rotary_white_pin), white_changed, CHANGE);

  pinMode(rotary_green_pin, INPUT_PULLUP);
  green_value = digitalRead(rotary_green_pin);

  delay(1000);

  rotary_index = 0;
}

void white_changed() {
  white_value = digitalRead(rotary_white_pin);
  green_value = digitalRead(rotary_green_pin);
  if (white_value == green_value) {
    rotary_index -= 1;
    clockwise = false;
  }
  else {
    rotary_index += 1;
    clockwise = true;
  }
}

void loop() {
  Serial.println("Index:  " + String(rotary_index));
  delay(1000);
}
