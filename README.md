# Learning CUDA
It's just a hobby project. I learn CUDA when I have some free time. Inspired by Udacity course "Intro to parallel programming" ([Youtube link](https://www.youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2 "Youtube link"))

# Prerequisites
The source code is using CUDA 10.1, VS2019 Community and Windows 10 SDK

# Nsight debugging
Nsight is an absolutely great tool, however, it is quite unstable when you try to debug your program on the same GPU being used to render OS UI. In my case, I have constantly had multiple total system freezes. Here are some useful tips on how to make it usable and stable.

Prerequisites:

- You might need a second GPU to avoid total system freeze while kernels debugging. An integrated one to your CPU suits well.
- While debugging, you might also need to disable the screen attached to the NVidia GPU. This makes the driver more stable because you release it from OS UI rendering. You don't have to unplug that monitor in case you have another one attached to the second GPU, just use Win + P hotkey shortcut and leave the second one.
- In the configuration described above, OpenGL-CUDA interoperability is impossible. This is caused by having different incompatible contexts on different GPUs. In this case, debugging your graphical program requires explicit data copying from CUDA to OpenGL memory spaces across a PCI-E bus.

Configuring the system:

- Mandatory: make sure you've set up "WDDM TDR Enabled" to **false** (Nsight Monitor -> General options). In some cases setting up "WDDM TDR Delay" may also help.
- In case of issues: if you experience total system freeze, setting up Visual Studio -> Nsight -> Options -> CUDA -> Preemption Preference to "Prefer Software Preempiton" may significantly stabilize your system. Each time you create a new solution this value is set to default, so you might to check it.

To start a debugging process use Visual Studio -> Nsight -> Start CUDA Debugging (Next-Gen) option only.