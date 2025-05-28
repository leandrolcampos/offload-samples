#include "OffloadUtils.hpp"
#include <iostream>

int main() {
  auto Devices = ols::getDevices();
  auto DeviceCount = Devices.size();

  printf("Detected %zu device(s), including the host.\n", DeviceCount);

  for (size_t DeviceId = 0; DeviceId < DeviceCount; DeviceId++) {
    const auto &CurrentDevice = Devices[DeviceId];
    auto Info = ols::getDeviceInfo(CurrentDevice);
    std::cout << "Device " << DeviceId << std::endl;
    std::cout << "    Name:           " << Info.Name << std::endl;
    std::cout << "    Vendor:         " << Info.Vendor << std::endl;
    std::cout << "    Driver Version: " << Info.DriverVersion << std::endl;
  }

  return 0;
}