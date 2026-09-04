from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["DeviceInput"]

type DeviceInput = int | str | list[int]


class _AcceleratorDeviceParser:
    def __init__(self, available_device_count: int, accelerator_name: str) -> None:
        if (
            isinstance(available_device_count, bool)
            or not isinstance(available_device_count, int)
            or available_device_count < 0
        ):
            raise ValueError("available_device_count must be a non-negative integer.")
        if not accelerator_name.strip():
            raise ValueError("accelerator_name must not be empty.")
        self._available_device_count: int = available_device_count
        self._accelerator_name: str = accelerator_name

    def parse(self, devices: DeviceInput) -> list[int]:
        normalized_devices: int | list[int]
        if isinstance(devices, bool):
            raise TypeError(self._type_error())
        if isinstance(devices, str):
            normalized_devices: int | list[int] = self._parse_string(devices)
        elif isinstance(devices, int):
            normalized_devices: int | list[int] = devices
        elif isinstance(devices, list):
            normalized_devices: int | list[int] = self._parse_identifiers(devices)
        else:
            raise TypeError(self._type_error())
        device_ids: list[int] = self._expand_count(normalized_devices, devices)
        self._validate_identifiers(device_ids, devices)
        return device_ids

    def _parse_string(self, devices: str) -> int | list[int]:
        stripped_devices: str = devices.strip()
        if not stripped_devices:
            raise TypeError(
                f"`devices` selected with {self._accelerator_name} cannot be empty."
            )
        if stripped_devices == "-1":
            return -1
        device_ids: list[int] = []
        device_token: str
        for device_token in stripped_devices.split(","):
            stripped_token: str = device_token.strip()
            if not stripped_token:
                continue
            try:
                device_ids.append(int(stripped_token))
            except ValueError as exception:
                raise TypeError(
                    f"`devices` selected with {self._accelerator_name} contains a "
                    "non-integer device ID."
                ) from exception
        return device_ids

    def _parse_identifiers(self, devices: list[int]) -> list[int]:
        device_id: int
        for device_id in devices:
            if type(device_id) is not int:
                raise TypeError(
                    f"`devices` selected with {self._accelerator_name} must contain "
                    "only integer device IDs."
                )
        return list(devices)

    def _expand_count(
        self,
        devices: int | list[int],
        original_devices: DeviceInput
    ) -> list[int]:
        if isinstance(devices, list):
            return devices
        if devices == -1:
            return list(range(self._available_device_count))
        if devices > 0:
            return list(range(devices))
        raise MisconfigurationError(
            f"`devices={original_devices!r}` is invalid for {self._accelerator_name}; "
            "expected -1 or a positive count."
        )

    def _validate_identifiers(
        self,
        device_ids: list[int],
        original_devices: DeviceInput
    ) -> None:
        if not device_ids:
            raise MisconfigurationError(
                f"`devices={original_devices!r}` selected no devices for "
                f"{self._accelerator_name}."
            )
        if len(device_ids) != len(set(device_ids)):
            raise MisconfigurationError("Device IDs must be unique.")
        unavailable_device_ids: list[int] = []
        device_id: int
        for device_id in device_ids:
            if device_id < 0:
                raise MisconfigurationError("Device IDs must be non-negative integers.")
            if device_id >= self._available_device_count:
                unavailable_device_ids.append(device_id)
        if unavailable_device_ids:
            available_device_ids: list[int] = list(
                range(self._available_device_count)
            )
            raise MisconfigurationError(
                f"Requested device IDs {device_ids} for {self._accelerator_name}, "
                f"but available IDs are {available_device_ids}."
            )

    def _type_error(self) -> str:
        return (
            f"`devices` selected with {self._accelerator_name} must be an int, "
            "a string, or a list of ints."
        )
