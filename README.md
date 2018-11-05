# Studying CUDA
It's just a hobby project. I study CUDA when I have some free time.

# Prerequesites
Source code is using CUDA 10.0, VS2017 Community and Windows 10 SDK 10.0.17134.0

# CUDA setup known issues
I've run into some issues during project setup. Here are some workarounds I've discovered.

## Compiler setup
### Removing a previous version of CUDA SDK
I had previous version of CUDA (9.2) which isn't compatible with VS2017. To use VS2017 I had to update CUDA SDK up to 10 version.

Before installing CUDA 10 SDK delete all previous SDK Toolsets and make sure you don't have "c:\Program Files\NVIDIA GPU Computing Toolkit" folder and any environment variables pointing on it, uninstallers may not remove them. System reboot is required to apply environment variables changes.

### Installing a new version of CUDA SDK
Just install it, but don't forget to reboot. Official installer doesn't suggest it, but this step is required to apply all environment variables properly. Otherwise Visual Studio won't find SDK location when you open an existing project or create a new one from scratch.

## Nsight debugging
Nsight is an absolutely great tool, however, it is quite unstable and buggy. Here are some useful tips how to make it usable.

- Make sure you've set up "WDDM TDR Enabled" to false (Nsight Monitor -> General options). In some cases setting up "WDDM TDR Delay" may also help.
- If you experience total system freeze, setting up Visual Studio -> Nsight -> Options -> CUDA -> Preemption Preference to "Prefer Software Preempiton" may significantly stabilize your system. Each time you create a new solution you have to check it.
- Use Visual Studio -> Nsight -> Start CUDA Debugging (Next-Gen) option only.

