# This module:
# 1. Exposes the structural Accelerator contract, built-in CPU, CUDA, and MPS
#    implementations, and accepted device-selection input
#
# Design decisions:
# - The package surface contains backend-local capabilities only; Trainer's
#   accelerator connector owns cross-component selection and compatibility
# - Concrete implementations inherit the Protocol explicitly so @override marks
#   every promised class- and instance-level operation
#
# Author: Rahul Sawhney

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.accelerators.cuda import CUDAAccelerator
from syntheticmind.accelerators.device_parser import DeviceInput
from syntheticmind.accelerators.mps import MPSAccelerator

__all__: list[str] = [
    "Accelerator",
    "CPUAccelerator",
    "CUDAAccelerator",
    "DeviceInput",
    "MPSAccelerator"
]
