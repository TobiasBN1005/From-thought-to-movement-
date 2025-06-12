#include "MPU6050_TBN.h"

// Konstruktor
MPU6050_TBN::MPU6050_TBN(uint8_t address, TwoWire* bus)
    : devAddr(address), wire(bus) {}

// Initialisering: wake up MPU6050
void MPU6050_TBN::initialize() {
    wire->begin();
    delay(10);  // Giv tid til opstart

    // Wake up MPU6050
    wire->beginTransmission(devAddr);
    wire->write(0x6B);  // Register: PWR_MGMT_1
    wire->write(0);     // Sæt til 0 for at vække enheden
    wire->endTransmission();

    delay(10);  // Stabiliseringstid
}

// Test om enheden svarer (returnerer true hvis OK)
bool MPU6050_TBN::testConnection() {
    wire->beginTransmission(devAddr);
    return (wire->endTransmission() == 0);
}

// Læs accelerometerdata (3 akser)
void MPU6050_TBN::getAcceleration(int16_t* ax, int16_t* ay, int16_t* az) {
    wire->beginTransmission(devAddr);
    wire->write(0x3B);  // ACCEL_XOUT_H
    wire->endTransmission(false);

    if (wire->requestFrom(devAddr, (uint8_t)6) == 6) {
        *ax = (wire->read() << 8) | wire->read();
        *ay = (wire->read() << 8) | wire->read();
        *az = (wire->read() << 8) | wire->read();
    } else {
        *ax = *ay = *az = -1;
    }
}

// Læs gyroskopdata (3 akser)
void MPU6050_TBN::getRotation(int16_t* gx, int16_t* gy, int16_t* gz) {
    wire->beginTransmission(devAddr);
    wire->write(0x43);  // GYRO_XOUT_H
    wire->endTransmission(false);

    if (wire->requestFrom(devAddr, (uint8_t)6) == 6) {
        *gx = (wire->read() << 8) | wire->read();
        *gy = (wire->read() << 8) | wire->read();
        *gz = (wire->read() << 8) | wire->read();
    } else {
        *gx = *gy = *gz = -1;
    }
}

// Læs både accelerometer og gyroskop (6 værdier)
void MPU6050_TBN::getMotion6(int16_t* ax, int16_t* ay, int16_t* az,
                             int16_t* gx, int16_t* gy, int16_t* gz) {
    wire->beginTransmission(devAddr);
    wire->write(0x3B);  // Start ved ACCEL_XOUT_H
    wire->endTransmission(false);

    if (wire->requestFrom(devAddr, (uint8_t)14) == 14) {
        *ax = (wire->read() << 8) | wire->read();
        *ay = (wire->read() << 8) | wire->read();
        *az = (wire->read() << 8) | wire->read();

        wire->read(); wire->read();  // spring temp over

        *gx = (wire->read() << 8) | wire->read();
        *gy = (wire->read() << 8) | wire->read();
        *gz = (wire->read() << 8) | wire->read();
    } else {
        *ax = *ay = *az = -1;
        *gx = *gy = *gz = -1;
    }
}
