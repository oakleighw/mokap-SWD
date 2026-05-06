#!/usr/bin/env python3
"""Test IC4 camera trigger signal reception and frame capture."""

import logging
import sys
from datetime import datetime
import imagingcontrol4 as ic4

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_trigger_capture():
    """Connect to IC4 camera, enable trigger mode, capture frames."""
    
    # Initialize IC4 library
    ic4.Library.init()
    
    # List available devices
    devices = ic4.DeviceEnum.devices()
    if not devices:
        logger.error("No IC4 devices found!")
        return
    
    logger.info(f"Found {len(devices)} IC4 camera(s)")
    
    # Use first device
    device_info = devices[0]
    logger.info(f"Using device: {device_info.model_name} (S/N: {device_info.serial_number})")
    
    # Create grabber
    grabber = ic4.Grabber()
    try:
        grabber.device = device_info
        logger.info("Device connected")
        
        # Get property map
        prop_map = grabber.device.property_map
        
        # Enable trigger mode
        logger.info("Enabling hardware trigger mode...")
        trigger_mode_prop = prop_map.find("TriggerMode")
        if trigger_mode_prop:
            trigger_mode_prop.value = "On"
            logger.info(f"TriggerMode set to: {trigger_mode_prop.value}")
        else:
            logger.error("TriggerMode property not found!")
            return
        
        # Set trigger activation polarity
        logger.info("Setting trigger activation edge...")
        try:
            trigger_activation_prop = prop_map.find("TriggerActivation")
            if trigger_activation_prop:
                # Try FallingEdge first (camera default)
                trigger_activation_prop.value = "FallingEdge"
                logger.info(f"TriggerActivation set to: {trigger_activation_prop.value}")
        except Exception as e:
            logger.warning(f"Could not set TriggerActivation: {e}")
        
        # Create SnapSink for frame capture
        snap_sink = ic4.SnapSink()
        
        # Start grabbing
        logger.info("Starting frame capture...")
        grabber.stream_setup(ic4.AcquisitionStart.ACQUISITION_START, snap_sink)
        
        # Try to capture frames
        logger.info("Waiting for trigger signals... (send signals from Pi)")
        logger.info("Press Ctrl+C to stop")
        
        capture_count = 0
        timeout_count = 0
        
        try:
            while True:
                try:
                    # Wait for frame with 3 second timeout
                    snap = snap_sink.snap_single(3000)
                    capture_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] Frame captured! Total: {capture_count}")
                    
                except ic4.IC4Exception as e:
                    timeout_count += 1
                    if "timeout" in str(e).lower():
                        logger.warning(f"Timeout waiting for trigger (#{timeout_count}). Is Pi sending signal?")
                    else:
                        logger.error(f"Capture error: {e}")
                        
        except KeyboardInterrupt:
            logger.info("\nStopping capture...")
            
        finally:
            grabber.stream_stop()
            logger.info(f"Captured {capture_count} frames, {timeout_count} timeouts")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        grabber.device = None
        logger.info("Device disconnected")
        ic4.Library.exit()


if __name__ == "__main__":
    test_trigger_capture()
