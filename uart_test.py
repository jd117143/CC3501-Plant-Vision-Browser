#!/usr/bin/env python3

import serial


SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200


def main() -> None:
    with serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1,
    ) as uart:
        print("Listening for RP2040 UART data...")

        while True:
            raw_line = uart.readline()

            if not raw_line:
                continue

            try:
                line = raw_line.decode(
                    "utf-8"
                ).strip()
            except UnicodeDecodeError:
                continue

            print(line)


if __name__ == "__main__":
    main()