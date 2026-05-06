#!/usr/bin/env python3
"""Simple Pi trigger pulse script - run on Raspberry Pi, press Enter to send trigger."""

import sys
import time

try:
    from gpiozero import OutputDevice
except ImportError:
    print("ERROR: gpiozero not installed. Install with: sudo pip install gpiozero")
    sys.exit(1)

# GPIO pin connected to Hirose trigger input (adjust to your pin number)
TRIGGER_PIN = 17  # Change this to your actual GPIO pin

def send_trigger():
    """Send a single trigger pulse."""
    trigger = OutputDevice(TRIGGER_PIN)
    
    try:
        print(f"Trigger pin: GPIO{TRIGGER_PIN}")
        print("Press Enter to send trigger pulse (Ctrl+C to exit)\n")
        
        pulse_count = 0
        while True:
            input()  # Wait for Enter key
            
            # Send pulse: LOW to HIGH to LOW (falling edge)
            trigger.off()
            time.sleep(0.001)  # 1ms low
            trigger.on()
            time.sleep(0.001)  # 1ms high
            trigger.off()
            
            pulse_count += 1
            print(f"Sent trigger pulse #{pulse_count}")
    
    except KeyboardInterrupt:
        print(f"\nSent {pulse_count} pulses total")
    
    finally:
        trigger.close()

if __name__ == "__main__":
    send_trigger()
