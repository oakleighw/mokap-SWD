import logging

import imagingcontrol4 as ic4
import numpy as np
from typing import Any, Dict, Optional, Tuple

from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


# Mapping from GenICam feature names to IC4 PropId constants
# Only include properties that actually exist in the current version of imagingcontrol4
def _build_feature_mapping():
    """Build feature mapping with fallback for missing properties"""
    mapping = {}
    
    # Define properties to try mapping
    properties_to_try = {
        'ExposureTime': 'EXPOSURE_TIME',
        'Gain': 'GAIN',
        'BlackLevel': 'BLACK_LEVEL',
        'Gamma': 'GAMMA',
        'AcquisitionMode': 'ACQUISITION_MODE',
        'ExposureAuto': 'EXPOSURE_AUTO',
        'GainAuto': 'GAIN_AUTO',
        'AcquisitionFrameRateEnable': 'ACQUISITION_FRAME_RATE_ENABLE',
        'AcquisitionFrameRate': 'ACQUISITION_FRAME_RATE',
        'ResultingFrameRate': 'RESULTING_FRAME_RATE',
        'Width': 'WIDTH',
        'Height': 'HEIGHT',
        'OffsetX': 'OFFSET_X',
        'OffsetY': 'OFFSET_Y',
        'OffsetAutoCenterEnable': 'OFFSET_AUTO_CENTER',
        'PixelFormat': 'PIXEL_FORMAT',
        'BinningHorizontal': 'BINNING_HORIZONTAL',
        'BinningVertical': 'BINNING_VERTICAL',
        'BinningHorizontalMode': 'BINNING_HORIZONTAL_MODE',
        'BinningVerticalMode': 'BINNING_VERTICAL_MODE',
        'TriggerSelector': 'TRIGGER_SELECTOR',
        'TriggerMode': 'TRIGGER_MODE',
        'TriggerSource': 'TRIGGER_SOURCE',
    }
    
    # Only add properties that exist
    for feature_name, prop_name in properties_to_try.items():
        try:
            mapping[feature_name] = getattr(ic4.PropId, prop_name)
        except AttributeError:
            logger.debug(f"IC4 PropId.{prop_name} not available in this version of imagingcontrol4")
    
    return mapping

FEATURE_MAPPING = _build_feature_mapping()


class IC4ImagingCamera(GenICamCamera):
    """
    Concrete implementation for The Imaging Source IC Imaging Control 4 cameras.
    Inherits all GenICam logic from the GenICamCamera parent class
    (only adds IC4-specific connection, grabbing, and feature access).
    """

    def __init__(self, device_info: ic4.DeviceInfo):
        """
        Args:
            device_info: DeviceInfo object from IC4 device enumeration
        """
        self._device_info = device_info
        self._grabber: Optional[ic4.Grabber] = None
        self._sink: Optional[ic4.SnapSink] = None
        self._warned_features = set()

        super().__init__(unique_id=device_info.serial)

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        """Override parent to gracefully handle missing trigger features in IC4"""
        if enabled:
            trigger_source = f"Line{''.join([char for char in str(self._trigger_line) if char.isdigit()])}"
            try:
                self._set_feature_value('TriggerSelector', 'FrameStart')
            except AttributeError as e:
                logger.debug(f"Could not set TriggerSelector: {e}")
            
            try:
                self._set_feature_value('TriggerMode', 'On')
            except AttributeError as e:
                logger.debug(f"Could not set TriggerMode: {e}")
            
            try:
                self._set_feature_value('TriggerSource', trigger_source)
            except AttributeError as e:
                logger.debug(f"Could not set TriggerSource: {e}")
            
            try:
                self._set_feature_value('AcquisitionFrameRateEnable', False)
            except AttributeError as e:
                logger.debug(f"Could not set AcquisitionFrameRateEnable: {e}")
        else:
            try:
                self._set_feature_value('TriggerMode', 'Off')
            except AttributeError as e:
                logger.debug(f"Could not set TriggerMode to Off: {e}")
            
            try:
                self._set_feature_value('AcquisitionFrameRateEnable', True)
            except AttributeError as e:
                logger.debug(f"Could not set AcquisitionFrameRateEnable to True: {e}")
        
        self._hardware_triggered = enabled
        self.framerate = self._framerate

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            logger.warning(f"Camera {self.unique_id} is already connected.")
            return

        try:
            # Library should already be initialized from discovery phase
            # Don't call ic4.Library.init() here to avoid "already called" error

            # Create and open grabber
            self._grabber = ic4.Grabber()
            self._grabber.device_open(self._device_info)
            self._is_connected = True

            self._apply_configuration(config)
            logger.info(f"Connected to IC Imaging camera {self.unique_id}")

        except ic4.IC4Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to IC Imaging camera {self.unique_id}: {e}") from e
        except Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Unexpected error connecting to IC Imaging camera {self.unique_id}: {e}") from e

    def disconnect(self) -> None:
        if self.is_grabbing:
            self.stop_grabbing()

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
                # Log current camera configuration for debugging
                try:
                    pix_fmt = self._get_feature_value('PixelFormat')
                    width = self._get_feature_value('Width')
                    height = self._get_feature_value('Height')
                    logger.debug(f"Camera {self.unique_id} config: PixelFormat={pix_fmt}, "
                               f"Width={width}, Height={height}")
                except Exception as e:
                    logger.debug(f"Could not log config for {self.unique_id}: {e}")

                logger.debug(f"Setting up stream for {self.unique_id} with SnapSink...")
                self._grabber.stream_setup(self._sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
                self._is_grabbing = True
                logger.debug(f"Started grabbing on {self.unique_id} - acquisition started")
            except ic4.IC4Exception as e:
                raise RuntimeError(f"Failed to start grabbing on {self.unique_id}: {e}") from e

    def stop_grabbing(self) -> None:
        if self.is_grabbing and self._grabber:
            try:
                self._grabber.stream_stop()
                self._is_grabbing = False
                logger.debug(f"Stopped grabbing on {self.unique_id}")
            except ic4.IC4Exception as e:
                logger.error(f"Error stopping grabbing on {self.unique_id}: {e}")

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._grabber or not self.is_connected:
            raise RuntimeError("Camera is not connected or has been released.")

        if not self.is_grabbing:
            raise RuntimeError("Camera is not grabbing. Call start_grabbing() first.")

        try:
            logger.debug(f"Calling snap_single with timeout={timeout_ms}ms on {self.unique_id}")
            image = self._sink.snap_single(timeout_ms)

            if image is None:
                raise IOError(f"Grab failed: Timeout after {timeout_ms} ms")

            logger.debug(f"Successfully grabbed frame from {self.unique_id}")

            # Convert IC4 image to numpy array using numpy_copy (safer than wrap)
            try:
                image_arr = image.numpy_copy()
            except AttributeError:
                # Fallback to numpy_wrap if numpy_copy not available
                try:
                    image_arr = image.numpy_wrap().copy()
                except AttributeError:
                    # Last resort: try to convert directly
                    image_arr = np.array(image)

            # Prepare metadata
            try:
                frame_meta = {
                    'timestamp': image.meta_data.device_timestamp_ns if hasattr(image, 'meta_data') else 0,
                    'frame_number': image.meta_data.device_frame_number if hasattr(image, 'meta_data') else 0,
                }
            except (AttributeError, TypeError):
                frame_meta = {'timestamp': 0, 'frame_number': 0}

            return image_arr, frame_meta

        except ic4.IC4Exception as e:
            logger.error(f"IC4 exception during snap_single on {self.unique_id}: code={e.code}, message={e.message}")
            raise IOError(f"Failed to grab frame: {e}") from e

    # --- GenICamCamera abstract contract ---

    def _get_nodemap(self):
        """
        IC4 doesn't have a nodemap concept like GenICam.
        This returns the PropertyMap instead.
        """
        if not self._grabber or not self.is_connected:
            raise RuntimeError("IC4 camera is not initialized.")

        return self._grabber.device_property_map

    def _get_feature_value(self, name: str) -> Any:
        try:
            # IC4 uses PropertyMap.find() to get typed property objects
            # then access the .value attribute
            mapped_name = self._map_feature_name(name)
            prop = self._grabber.device_property_map.find(mapped_name)
            return prop.value
        except ic4.IC4Exception as e:
            if name == 'ResultingFrameRate':
                try:
                    return self._get_feature_value('AcquisitionFrameRate')
                except:
                    return 60.0  # Last resort fallback
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e
        except (KeyError, IndexError, AttributeError) as e:
            raise AttributeError(f"Feature '{name}' not found: {e}") from e

    def _set_feature_value(self, name: str, value: Any) -> Any:
        try:
            mapped_name = self._map_feature_name(name)

            #icImaging cameras don't have thsi toggle
            if name == 'AcquisitionFrameRateEnable':
                try:
                    self._grabber.device_property_map.set_value(mapped_name, value)
                except ic4.IC4Exception:
                    # log if  haven't logged this specific warning yet
                    if name not in self._warned_features:
                        logger.debug(f"Camera does not have an explicit '{name}' switch. Proceeding anyway.")
                        self._warned_features.add(name)
                return True
            
            self._grabber.device_property_map.set_value(mapped_name, value)

            prop = self._grabber.device_property_map.find(mapped_name)
            return prop.value
        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to set feature '{name}' to '{value}': {e}") from e
        except (KeyError, IndexError, AttributeError) as e:
            raise AttributeError(f"Feature '{name}' not found: {e}") from e

    def _get_feature_min_value(self, name: str) -> Any:
        try:
            mapped = self._map_feature_name(name)
            prop = self._grabber.device_property_map.find(mapped)
            return getattr(prop, 'minimum', 0) # Fallback to 0
        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get min for feature '{name}': {e}") from e
        except (KeyError, IndexError, AttributeError) as e:
            raise AttributeError(f"Feature '{name}' not found or has no min value: {e}") from e
        except Exception:
            return 0

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            mapped = self._map_feature_name(name)
            prop = self._grabber.device_property_map.find(mapped)
            # Default to 1000 for framerate/width/height if max is missing
            return getattr(prop, 'maximum', 1000) 
        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e
        except (KeyError, IndexError, AttributeError) as e:
            raise AttributeError(f"Feature '{name}' not found or has no max value: {e}") from e
        except Exception:
            return 1000

    def _get_feature_entries(self, name: str) -> list[str]:
        """
        Get enumeration entries for a feature.
        Returns a list of available values for enumeration properties.
        """
        try:
            # For enumerations, find the property and get entries
            mapped_name = self._map_feature_name(name)
            prop = self._grabber.device_property_map.find(mapped_name)

            # IC4 PropEnumeration has .entries attribute with PropEnumEntry objects
            if hasattr(prop, 'entries') and prop.entries:
                return [entry.name for entry in prop.entries]
            else:
                logger.debug(f"No valid values found for feature '{name}'")
                return []

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get entries for feature '{name}': {e}") from e
        except (KeyError, IndexError, AttributeError) as e:
            raise AttributeError(f"Feature '{name}' not found or is not enumeration: {e}") from e

    def _map_feature_name(self, name: str) -> Any: # Changed from ic4.PropId to Any
        """
        Map GenICam feature names to IC4 PropId constants or fallback strings.
        """
        # 1. Check the manual mapping first
        if name in FEATURE_MAPPING:
            return FEATURE_MAPPING[name]

        # 2. Try to find the PropId attribute directly
        try:
            return getattr(ic4.PropId, name)
        except AttributeError:
            pass

        # 3. Try common UPPER_SNAKE_CASE conversion (e.g. PixelFormat -> PIXEL_FORMAT)
        try:
            upper_name = ''.join(['_' + c if c.isupper() else c for c in name]).lstrip('_').upper()
            return getattr(ic4.PropId, upper_name)
        except AttributeError:
            pass

        # 4. Final Fallback: Return the raw string
        # IC4's property_map.find() can take a string name directly
        return name
        

        # 3. Final Fallback: Return the name as a string 
        # IC4 find() often accepts the literal GenICam string
        return name
        # # Try converting CamelCase to UPPER_CASE format
        # try:
        #     upper_name = ''.join(['_' + c if c.isupper() else c for c in name]).lstrip('_').upper()
        #     prop_id = getattr(ic4.PropId, upper_name)
        #     return prop_id
        # except AttributeError:
        #     raise AttributeError(f"Feature '{name}' not found in IC4 PropId mapping")
        


    # --- IC4-specific property overrides for graceful handling of missing features ---

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int):
        """
        Set binning. IC4 cameras may not support binning features,
        so we gracefully skip them if not available.
        """
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        # Try to set binning, but don't fail if not supported
        h_val = value
        v_val = value
        try:
            h_val = self._set_feature_value('BinningHorizontal', value)
        except AttributeError as e:
            logger.debug(f"BinningHorizontal not supported by this camera: {e}")
        
        try:
            v_val = self._set_feature_value('BinningVertical', value)
        except AttributeError as e:
            logger.debug(f"BinningVertical not supported by this camera: {e}")

        self._binning = h_val

        # Always set ROI to full resolution when binning changes
        try:
            self._set_feature_value('OffsetX', 0)
            self._set_feature_value('OffsetY', 0)
            width = self._get_feature_max_value('Width')
            height = self._get_feature_max_value('Height')
            self._set_feature_value('Width', width)
            self._set_feature_value('Height', height)
            self._roi = (0, 0, width, height)
        except AttributeError as e:
            logger.warning(f"Could not set ROI: {e}")

        if was_grabbing:
            self.start_grabbing()

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning_mode.setter
    def binning_mode(self, value: str):
        """
        Set binning mode. IC4 cameras may not support this feature,
        so we gracefully skip it if not available.
        """
        mode = 'Average' if value.lower() in ['a', 'avg', 'average'] else 'Sum'
        
        # Check if features exist in the property map to avoid driver errors
        props = self._grabber.device_property_map
        if 'BinningHorizontalMode' not in props:
            if 'BinningHorizontalMode' not in self._warned_features:
                logger.debug(f"BinningHorizontalMode not supported by this camera. Skipping.")
                self._warned_features.add('BinningHorizontalMode')
            self._binning_mode = 'Sum'
            return

        try:
            self._binning_mode = self._set_feature_value('BinningHorizontalMode', mode)
            self._set_feature_value('BinningVerticalMode', mode)
        except Exception as e:
            # Catching the case where it exists but is currently read-only
            pass

    @property
    def available_binning_modes(self) -> list[str]:
        """
        Get available binning modes. Returns defaults if not supported.
        """
        try:
            return self._get_feature_entries('BinningHorizontalMode')
        except AttributeError as e:
            logger.debug(f"Cannot query available binning modes: {e}")
            return ['Sum', 'Average']  # Default fallback
