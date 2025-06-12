#ifndef MPU6050_TBN_H
#define MPU6050_TBN_H

#include <Arduino.h>
#include <Wire.h>

class MPU6050_TBN {
public:
    // Konstruktor: Vælg I2C-adresse og bus (default: 0x68 og Wire)
    MPU6050_TBN(uint8_t address = 0x68, TwoWire* bus = &Wire);

    // Initialisering og forbindelsestest
    void initialize();
    bool testConnection();

    // Læs accelerometerdata (x, y, z)
    void getAcceleration(int16_t* ax, int16_t* ay, int16_t* az);

    // Læs gyroskopdata (x, y, z)
    void getRotation(int16_t* gx, int16_t* gy, int16_t* gz);

    // Læs både accelerometer og gyroskop (6 værdier)
    void getMotion6(int16_t* ax, int16_t* ay, int16_t* az,
                    int16_t* gx, int16_t* gy, int16_t* gz);

private:
    uint8_t devAddr;
    TwoWire* wire;
};

#endif
