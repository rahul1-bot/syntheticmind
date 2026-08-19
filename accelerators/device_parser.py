from syntheticmind.utilities.exceptions import MisconfigurationError

type DeviceInput = int | str | list[int]


def parse_gpu_devices(
    devices: DeviceInput,
    available_device_count: int,
    accelerator_name: str
) -> list[int]:
    normalized_devices: int | list[int]
    if isinstance(devices, bool):
        raise TypeError(
            f"`devices` selected with {accelerator_name} must be an int, a string, or a list of ints."
        )
    if isinstance(devices, str):
        stripped_devices: str = devices.strip()
        if not stripped_devices:
            raise TypeError(
                f"`devices` selected with {accelerator_name} cannot be an empty string."
            )
        if "," in stripped_devices:
            try:
                normalized_devices = [
                    int(device_id.strip())
                    for device_id in stripped_devices.split(",")
                    if device_id.strip()
                ]
            except ValueError as exception:
                raise TypeError(
                    f"`devices` selected with {accelerator_name} contains a non-integer device ID."
                ) from exception
        else:
            try:
                normalized_devices = int(stripped_devices)
            except ValueError as exception:
                raise TypeError(
                    f"`devices` selected with {accelerator_name} must be numeric or comma-separated IDs."
                ) from exception
    elif isinstance(devices, int):
        normalized_devices = devices
    elif isinstance(devices, list):
        if any(type(device_id) is not int for device_id in devices):
            raise TypeError(
                f"`devices` selected with {accelerator_name} must contain only integer device IDs."
            )
        normalized_devices = list(devices)
    else:
        raise TypeError(
            f"`devices` selected with {accelerator_name} must be an int, a string, or a list of ints."
        )

    if isinstance(normalized_devices, int):
        if normalized_devices == -1:
            device_ids: list[int] = list(range(available_device_count))
        elif normalized_devices > 0:
            device_ids = list(range(normalized_devices))
        else:
            raise MisconfigurationError(
                f"`devices={devices!r}` is invalid for {accelerator_name}; expected -1 or a positive count."
            )
    else:
        device_ids = normalized_devices

    if not device_ids:
        raise MisconfigurationError(
            f"`devices={devices!r}` selected no devices for {accelerator_name}."
        )
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationError("Device IDs must be unique.")
    if any(device_id < 0 for device_id in device_ids):
        raise MisconfigurationError("Device IDs must be non-negative integers.")
    available_device_ids: list[int] = list(range(available_device_count))
    unavailable_device_ids: list[int] = [
        device_id
        for device_id in device_ids
        if device_id not in available_device_ids
    ]
    if unavailable_device_ids:
        raise MisconfigurationError(
            f"Requested device IDs {device_ids} for {accelerator_name}, but available IDs are "
            f"{available_device_ids}."
        )
    return device_ids
