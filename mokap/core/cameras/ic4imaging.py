import logging

import imagingcontrol4 as ic4
import numpy as np
from typing import Any, Dict, Optional, Tuple

from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


def _build_feature_mapping():
    """Build mapping from GenICam names to IC4 PropId constants"""
    mapping = {}
    properties_to_try = {
        'ExposureTime': 'EXPOSURE_TIME',
        'Gain': 'GAIN',
        'BlackLevel': 'BLACK_LEVEL',
        'Gamma': 'GAMMA',
        'AcquisitionMode': 'ACQUISITION_MODE',
        'ExposureAuto': 'EXPOSURE_AUTO',
        'GainAuto': 'GAIN_AUTO',
        # Note: IC4 does not have AcquisitionFrameRateEnable - it uses AcquisitionMode instead
        'AcquisitionFrameRate': 'ACQUISITION_FRAME_RATE',
        'ResultingFrameRate': 'RESULTING_FRAME_RATE',
        'Width': 'WIDTH',
        'Height': 'HEIGHT',
        'OffsetX': 'OFFSET_X',
        'OffsetY': 'OFFSET_Y',
        'PixelFormat': 'PIXEL_FORMAT',
        'BinningHorizontal': 'BINNING_HORIZONTAL',
        'BinningVertical': 'BINNING_VERTICAL',
        'BinningHorizontalMode': 'BINNING_HORIZONTAL_MODE',
        'BinningVerticalMode': 'BINNING_VERTICAL_MODE',
        'TriggerSelector': 'TRIGGER_SELECTOR',
        'TriggerMode': 'TRIGGER_MODE',
        'TriggerSource': 'TRIGGER_SOURCE',
    }
    
    for feature_name, prop_name in properties_to_try.items():
        try:
            mapping[feature_name] = getattr(ic4.PropId, prop_name)
        except AttributeError:
            logger.debug(f"IC4 PropId.{prop_name} not available in imagingcontrol4")
    
    return mapping


FEATURE_MAPPING = _build_feature_mapping()


class IC4ImagingCamera(GenICamCamera):
    """
    Implementation for The Imaging Source IC Imaging Control 4 cameras
    (only adds IC4-specific connection, grabbing, and feature access)
    """

    def __init__(self, device_info: ic4.DeviceInfo):
        self._device_info = device_info
        self._grabber: Optional[ic4.Grabber] = None
        self._sink: Optional[ic4.SnapSink] = None
        self._warned_features = set()
        self._actual_max_framerate: Optional[float] = None # may change based on resolution/bandwidth, so we probe it at connection time and cache the result

        super().__init__(unique_id=device_info.serial)

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            logger.warning(f"Camera {self.unique_id} is already connected.")
            return
        try:
            self._grabber = ic4.Grabber()
            self._grabber.device_open(self._device_info)
            self._is_connected = True
            self._apply_configuration(config)
            self._probe_actual_max_framerate()

            logger.info(f"Connected to IC Imaging camera {self.unique_id}")

        except ic4.IC4Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to IC Imaging camera {self.unique_id}: {e}") from e

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()
        if self._grabber:
            try:
                self._grabber.device_close()
            except ic4.IC4Exception as e:
                logger.warning(f"Error closing IC4 device: {e}")
        self._grabber = None
        self._sink = None
        self._is_connected = False

        logger.info(f"Disconnected from IC Imaging camera {self.unique_id}")

    def start_grabbing(self) -> None:
        if self.is_connected and not self.is_grabbing:
            if not self._sink:
                self._sink = ic4.SnapSink()

            try:
                self._grabber.stream_setup(self._sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
                self._is_grabbing = True
            except ic4.IC4Exception as e:
                raise RuntimeError(f"Failed to start grabbing on {self.unique_id}: {e}") from e

    def stop_grabbing(self) -> None:
        if self.is_grabbing and self._grabber:
            try:
                self._grabber.stream_stop()
                self._is_grabbing = False
            except ic4.IC4Exception as e:
                logger.error(f"Error stopping grabbing on {self.unique_id}: {e}")

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._grabber or not self.is_connected:
            raise RuntimeError("Camera is not connected or has been released.")

        if not self.is_grabbing:
            raise RuntimeError("Camera is not grabbing. Call start_grabbing() first.")

        if self.hardware_triggered:
            timeout_ms = max(timeout_ms, 5000)

        try:
            image = self._sink.snap_single(timeout_ms)

            if image is None:
                raise IOError(f"Grab failed: Timeout after {timeout_ms} ms")

            try:
                image_arr = image.numpy_copy()
            except AttributeError:
                image_arr = image.numpy_wrap().copy()

            try:
                frame_meta = {
                    'timestamp': image.meta_data.device_timestamp_ns if hasattr(image, 'meta_data') else 0,
                    'frame_number': image.meta_data.device_frame_number if hasattr(image, 'meta_data') else 0,
                }
            except (AttributeError, TypeError):
                frame_meta = {'timestamp': 0, 'frame_number': 0}

            return image_arr, frame_meta

        except ic4.IC4Exception as e:
            raise IOError(f"Failed to grab frame: {e}") from e

    # --- GenICamCamera abstract contract ---

    def _get_feature_value(self, name: str) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)

            if prop is None:
                raise AttributeError(f"Feature '{name}' not found")

            return prop.value

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e

    def _set_feature_value(self, name: str, value: Any) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            self._grabber.device_property_map.set_value(prop_id, value)

            prop = self._grabber.device_property_map.find(prop_id)
            return prop.value if prop else value

        except ic4.IC4Exception as e:
            if name not in self._warned_features:
                logger.debug(f"Feature '{name}' not available or not writable: {e}")
                self._warned_features.add(name)
            return value

    def _get_feature_entries(self, name: str) -> list[str]:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)

            if prop is None or not hasattr(prop, 'entries'):
                return []

            return [entry.name for entry in prop.entries]

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get entries for feature '{name}': {e}") from e

    def _get_feature_min_value(self, name: str) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)
            
            if prop is None:
                logger.debug(f"Property '{name}' not found for min value query")
                return 0
            
            if not hasattr(prop, 'minimum'):
                logger.debug(f"Property '{name}' has no 'minimum' attribute")
                return 0

            return prop.minimum

        except ic4.IC4Exception as e:
            logger.debug(f"IC4 exception getting min for feature '{name}': {e}")
            return 0

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            if name == 'AcquisitionFrameRate' and self._actual_max_framerate is not None:
                return self._actual_max_framerate

            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)
            
            if prop is None:
                logger.debug(f"Property '{name}' not found for max value query")
                return 1000
            
            if not hasattr(prop, 'maximum'):
                logger.debug(f"Property '{name}' has no 'maximum' attribute")
                return 1000

            return prop.maximum

        except ic4.IC4Exception as e:
            logger.debug(f"IC4 exception getting max for feature '{name}': {e}")
            return 1000

    def _probe_actual_max_framerate(self) -> None:
        """Probe actual max framerate at current resolution (bandwidth-limited)"""
        try:
            theoretical_max = float(self._get_feature_max_value('AcquisitionFrameRate'))
            current_fps = float(self._get_feature_value('AcquisitionFrameRate'))

            self._set_feature_value('AcquisitionFrameRate', theoretical_max)
            actual_max = float(self._get_feature_value('AcquisitionFrameRate'))

            self._set_feature_value('AcquisitionFrameRate', current_fps)

            self._actual_max_framerate = actual_max
            logger.debug(f"Probed actual max fps: {actual_max} (theoretical: {theoretical_max})")

        except Exception as e:
            logger.debug(f"Could not probe actual max framerate: {e}")
            self._actual_max_framerate = None

    def _get_prop_id(self, name: str) -> Any:
        """Convert GenICam feature name to IC4 PropId (tries: mapping → attribute → UPPER_SNAKE_CASE → string)"""
        if name in FEATURE_MAPPING:
            return FEATURE_MAPPING[name]

        try:
            return getattr(ic4.PropId, name)
        except AttributeError:
            pass

        try:
            upper_name = ''.join(['_' + c if c.isupper() else c for c in name]).lstrip('_').upper()
            return getattr(ic4.PropId, upper_name)
        except AttributeError:
            pass

        return name

    # --- IC4-specific property overrides ---

    @property
    def framerate(self) -> float:
        try:
            self._framerate = float(self._get_feature_value('AcquisitionFrameRate'))
        except AttributeError:
            pass
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):
        try:
            if not self.hardware_triggered:
                try:
                    self._set_feature_value('AcquisitionMode', 'Continuous')
                except AttributeError:
                    pass

            min_fps, max_fps = self.framerate_range
            clamped_value = max(min_fps, min(value, max_fps))

            actual_value = self._set_feature_value('AcquisitionFrameRate', clamped_value)
            self._framerate = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support framerate control: {e}")
            self._framerate = 0.0

    @property
    def framerate_range(self) -> Tuple[float, float]:
        try:
            min_fps = float(self._get_feature_min_value('AcquisitionFrameRate'))
            max_fps = self._actual_max_framerate if self._actual_max_framerate else float(self._get_feature_max_value('AcquisitionFrameRate'))
            return min_fps, max_fps

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine framerate range for {self.unique_id}")
            return 0.5, 500.0

    @property
    def exposure(self) -> float:
        try:
            self._exposure = float(self._get_feature_value('ExposureTime'))
        except AttributeError:
            pass
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        try:
            min_exp = float(self._get_feature_min_value('ExposureTime'))
            max_exp = float(self._get_feature_max_value('ExposureTime'))
            clamped_value = max(min_exp, min(value, max_exp))

            actual_value = self._set_feature_value('ExposureTime', clamped_value)
            self._exposure = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support exposure control: {e}")
            self._exposure = 5000.0

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def exposure_range(self) -> Tuple[float, float]:
        try:
            min_exp = float(self._get_feature_min_value('ExposureTime'))
            max_exp = float(self._get_feature_max_value('ExposureTime'))
            return min_exp, max_exp

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine exposure range for {self.unique_id}")
            return 1.0, 1000000.0

    @property
    def gain(self) -> float:
        try:
            self._gain = float(self._get_feature_value('Gain'))
        except AttributeError:
            pass
        return self._gain

    @gain.setter
    def gain(self, value: float):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        try:
            min_gain = float(self._get_feature_min_value('Gain'))
            max_gain = float(self._get_feature_max_value('Gain'))
            clamped_value = max(min_gain, min(value, max_gain))

            actual_value = self._set_feature_value('Gain', clamped_value)
            self._gain = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support gain control: {e}")
            self._gain = 1.0

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def gain_range(self) -> Tuple[float, float]:
        try:
            min_gain = float(self._get_feature_min_value('Gain'))
            max_gain = float(self._get_feature_max_value('Gain'))
            return min_gain, max_gain

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine gain range for {self.unique_id}")
            return 0.0, 32.0

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        """
        Override parent to handle IC4's trigger configuration.
        IC4 cameras don't have TriggerSource - trigger input is fixed to the Hirose connector.
        We just enable/disable TriggerMode and set activation edge.
        """
        if enabled:
            try:
                # Set which trigger to use (only FrameStart available)
                self._set_feature_value('TriggerSelector', 'FrameStart')
            except AttributeError:
                logger.debug("TriggerSelector not available")

            try:
                # Enable trigger mode
                self._set_feature_value('TriggerMode', 'On')
                logger.debug(f"Enabled trigger mode on {self.unique_id}")
            except AttributeError as e:
                logger.error(f"Cannot enable TriggerMode: {e}")
                self._hardware_triggered = False
                return

            try:
                # Try FallingEdge first (camera default), then fallback to RisingEdge
                # Both work with PWM since we generate both edges, but must match camera's expectation
                actual_value = None
                
                try:
                    self._set_feature_value('TriggerActivation', 'FallingEdge')
                    # Verify what was actually set
                    actual_value = self._get_feature_value('TriggerActivation')
                    logger.debug(f"TriggerActivation set request: FallingEdge → actual value: {actual_value}")
                except Exception as e1:
                    logger.debug(f"FallingEdge not available: {e1}, trying RisingEdge")
                    try:
                        self._set_feature_value('TriggerActivation', 'RisingEdge')
                        actual_value = self._get_feature_value('TriggerActivation')
                        logger.debug(f"TriggerActivation set request: RisingEdge → actual value: {actual_value}")
                    except Exception as e2:
                        logger.warning(f"Neither FallingEdge nor RisingEdge available: {e1} / {e2}")
                        actual_value = None
                
                if actual_value is None:
                    logger.warning("Could not set TriggerActivation. Camera will use its default (likely FallingEdge)")
                        
            except AttributeError:
                logger.debug("TriggerActivation not available, using camera default")

            self._hardware_triggered = True
            logger.info(f"Hardware trigger enabled on {self.unique_id}")

        else:
            try:
                self._set_feature_value('TriggerMode', 'Off')
                logger.debug(f"Disabled trigger mode on {self.unique_id}")
            except AttributeError:
                logger.debug("TriggerMode disable failed")
            
            self._hardware_triggered = False
