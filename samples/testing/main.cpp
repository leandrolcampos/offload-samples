#include "DeviceContext.hpp"
#include <iostream>

using namespace testing;

static void devices() {
  std::cout << "--- DEVICE ---\n\n"
            << "Number of devices: " << countDevices() << "\n";

  if (countDevices() > 0) {
    DeviceContext Context;
    std::cout << "  DeviceId: 0\n"
              << "  Name:     " << Context.getName() << "\n"
              << "  Platform: " << Context.getPlatform() << "\n\n";
  }
}

static void helloWorld() {
  std::cout << "--- HELLO WORLD ---\n\n";

  const std::string DeviceBinsDirectory = DEVICE_CODE_PATH;
  DeviceContext Context;
  auto Image = Context.loadBinary(DeviceBinsDirectory, "HelloWorld");
  auto Kernel = Context.getKernel<void()>(Image, "printHelloWorld");
  Context.launchKernel(Kernel, 4, 2);

  std::cout << "\n\n";
}

int main() {
  devices();
  helloWorld();

  return 0;
}