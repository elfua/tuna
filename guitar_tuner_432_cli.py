#!/usr/bin/env python3
"""
Guitar Tuner 432 Hz - Command Line Application

Plays guitar/guitalele strings for tuning with A4=432Hz (or custom frequency).
Supports multiple audio backends with automatic fallback.

Usage:
    python3 guitar_tuner_432.py [--frequency FREQ] [--instrument TYPE]
    
Examples:
    python3 guitar_tuner_432.py
    python3 guitar_tuner_432.py --frequency 440
    python3 guitar_tuner_432.py --instrument guitalele
    python3 guitar_tuner_432.py --frequency 440 --instrument guitalele
"""

import sys
import argparse
import math
import time
import platform
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from pathlib import Path

# Python version check
if sys.version_info < (3, 7):
    print("ERROR: This script requires Python 3.7 or higher")
    print(f"Current version: {sys.version}")
    sys.exit(1)


# ============================================================================
# INSTRUMENT CONFIGURATION
# ============================================================================

@dataclass
class GuitarString:
    """Represents a single guitar string with its tuning information."""
    note: str
    octave: int
    semitones: int  # Semitones from A4 (negative = lower, positive = higher)
    
    def __str__(self) -> str:
        return f"{self.note}{self.octave}"
    
    def calculate_frequency(self, base_freq: float = 432.0) -> float:
        """
        Calculate frequency using equal temperament.
        
        Formula: f(n) = f0 × 2^(n/12)
        where f0 is the reference frequency (A4) and n is semitones from A4.
        """
        return base_freq * math.pow(2, self.semitones / 12)


@dataclass
class Instrument:
    """Represents a stringed instrument with its tuning configuration."""
    name: str
    strings: List[GuitarString]  # Ordered from highest to lowest
    
    def get_all_strings(self) -> List[GuitarString]:
        """Return strings in order from highest to lowest."""
        return self.strings


# Instrument definitions
INSTRUMENTS = {
    'guitar': Instrument(
        name='Classic Guitar',
        strings=[
            GuitarString('E', 4, -5),   # 1st string (highest): 323.63 Hz
            GuitarString('B', 3, -10),  # 2nd string: 242.45 Hz
            GuitarString('G', 3, -14),  # 3rd string: 192.43 Hz
            GuitarString('D', 3, -19),  # 4th string: 144.16 Hz
            GuitarString('A', 2, -24),  # 5th string: 108.00 Hz
            GuitarString('E', 2, -29),  # 6th string (lowest): 80.91 Hz
        ]
    ),
    'guitalele': Instrument(
        name='Guitalele',
        strings=[
            GuitarString('A', 4, 0),    # 1st string (highest): 432.00 Hz
            GuitarString('E', 4, -5),   # 2nd string: 323.63 Hz
            GuitarString('C', 4, -9),   # 3rd string: 256.87 Hz
            GuitarString('G', 3, -14),  # 4th string: 192.43 Hz
            GuitarString('D', 3, -19),  # 5th string: 144.16 Hz
            GuitarString('A', 2, -24),  # 6th string (lowest): 108.00 Hz
        ]
    )
}


# ============================================================================
# AUDIO GENERATION
# ============================================================================

def generate_tone(frequency: float, duration: float = 5.0, 
                  sample_rate: int = 44100) -> Tuple[bytes, int]:
    """
    Generate a sine wave tone with exponential attack/release envelopes.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    import numpy as np
    
    # Generate time array
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t, dtype=np.float32)
    
    # Apply exponential envelope for smooth attack/release
    attack_time = 0.05  # 50ms
    release_time = 0.1   # 100ms
    
    attack_samples = int(sample_rate * attack_time)
    release_samples = int(sample_rate * release_time)
    
    # Exponential attack (0.001 to 0.3)
    if attack_samples > 0:
        attack_envelope = np.logspace(
            np.log10(0.001), 
            np.log10(0.3), 
            attack_samples, 
            dtype=np.float32
        )
        audio[:attack_samples] *= attack_envelope
    
    # Sustain (0.3)
    sustain_start = attack_samples
    sustain_end = num_samples - release_samples
    if sustain_end > sustain_start:
        audio[sustain_start:sustain_end] *= 0.3
    
    # Exponential release (0.3 to 0.001)
    if release_samples > 0:
        release_envelope = np.logspace(
            np.log10(0.3), 
            np.log10(0.001), 
            release_samples, 
            dtype=np.float32
        )
        audio[-release_samples:] *= release_envelope
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes(), sample_rate


# ============================================================================
# AUDIO BACKENDS
# ============================================================================

class AudioBackend:
    """Base class for audio backends."""
    
    name: str = "base"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available."""
        raise NotImplementedError
    
    @classmethod
    def play_tone(cls, frequency: float, duration: float) -> bool:
        """Play a tone. Returns True on success, False on failure."""
        raise NotImplementedError
    
    @classmethod
    def get_install_instructions(cls) -> str:
        """Get installation instructions for this backend."""
        return "No installation instructions available."


class SoundDeviceBackend(AudioBackend):
    """sounddevice backend (PortAudio wrapper, cross-platform, modern)."""
    
    name = "sounddevice"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import sounddevice
            return True
        except ImportError:
            return False
    
    @classmethod
    def play_tone(cls, frequency: float, duration: float) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
            
            audio_bytes, sample_rate = generate_tone(frequency, duration)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Play audio
            sd.play(audio_array, sample_rate)
            sd.wait()
            
            return True
        except Exception as e:
            print(f"  ⚠️  sounddevice error: {e}")
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return """
sounddevice (recommended - cross-platform, modern):
  pip install sounddevice numpy
  
Note: Requires PortAudio library:
  - macOS: brew install portaudio
  - Linux: sudo apt-get install libportaudio2 (Debian/Ubuntu)
          sudo dnf install portaudio (Fedora)
  - Windows: Usually works out of the box
"""


class PyAudioBackend(AudioBackend):
    """PyAudio backend (PortAudio wrapper, cross-platform, older but stable)."""
    
    name = "pyaudio"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import pyaudio
            return True
        except ImportError:
            return False
    
    @classmethod
    def play_tone(cls, frequency: float, duration: float) -> bool:
        try:
            import pyaudio
            
            audio_bytes, sample_rate = generate_tone(frequency, duration)
            
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            # Play audio
            stream.write(audio_bytes)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return True
        except Exception as e:
            print(f"  ⚠️  pyaudio error: {e}")
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return """
pyaudio (cross-platform, stable):
  pip install pyaudio numpy
  
Note: Requires PortAudio library:
  - macOS: brew install portaudio
  - Linux: sudo apt-get install portaudio19-dev python3-pyaudio (Debian/Ubuntu)
          sudo dnf install portaudio-devel (Fedora)
  - Windows: pip install pipwin && pipwin install pyaudio
"""


class SimpleAudioBackend(AudioBackend):
    """simpleaudio backend (pure Python, lightweight)."""
    
    name = "simpleaudio"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import simpleaudio
            return True
        except ImportError:
            return False
    
    @classmethod
    def play_tone(cls, frequency: float, duration: float) -> bool:
        try:
            import simpleaudio as sa
            
            audio_bytes, sample_rate = generate_tone(frequency, duration)
            
            # Play audio
            play_obj = sa.play_buffer(audio_bytes, 1, 2, sample_rate)
            play_obj.wait_done()
            
            return True
        except Exception as e:
            print(f"  ⚠️  simpleaudio error: {e}")
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        return """
simpleaudio (lightweight, pure Python):
  pip install simpleaudio numpy
  
Note: Requires system audio libraries:
  - macOS: Should work out of the box
  - Linux: sudo apt-get install python3-dev libasound2-dev (Debian/Ubuntu)
  - Windows: Should work out of the box
"""


class SystemCommandBackend(AudioBackend):
    """Fallback to system command-line audio tools."""
    
    name = "system_command"
    
    @classmethod
    def is_available(cls) -> bool:
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            return cls._command_exists('afplay')
        elif system == 'Linux':
            return (cls._command_exists('aplay') or 
                   cls._command_exists('paplay') or 
                   cls._command_exists('ffplay'))
        elif system == 'Windows':
            return True  # PowerShell is always available
        
        return False
    
    @staticmethod
    def _command_exists(command: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run(
                ['which', command] if platform.system() != 'Windows' else ['where', command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @classmethod
    def play_tone(cls, frequency: float, duration: float) -> bool:
        try:
            import numpy as np
            
            # Generate audio
            audio_bytes, sample_rate = generate_tone(frequency, duration)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_path = tmp_file.name
                cls._write_wav_file(wav_path, audio_bytes, sample_rate)
            
            try:
                # Play using system command
                success = cls._play_wav_file(wav_path)
                return success
            finally:
                # Cleanup
                try:
                    os.unlink(wav_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"  ⚠️  system command error: {e}")
            return False
    
    @staticmethod
    def _write_wav_file(filepath: str, audio_bytes: bytes, sample_rate: int):
        """Write a simple WAV file."""
        import struct
        
        with open(filepath, 'wb') as f:
            # WAV header
            num_channels = 1
            sample_width = 2  # 16-bit
            num_samples = len(audio_bytes) // sample_width
            
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(audio_bytes)))
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # chunk size
            f.write(struct.pack('<H', 1))   # PCM format
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', sample_rate * num_channels * sample_width))
            f.write(struct.pack('<H', num_channels * sample_width))
            f.write(struct.pack('<H', sample_width * 8))
            
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', len(audio_bytes)))
            f.write(audio_bytes)
    
    @classmethod
    def _play_wav_file(cls, filepath: str) -> bool:
        """Play WAV file using system-specific command."""
        system = platform.system()
        
        try:
            if system == 'Darwin':  # macOS
                subprocess.run(['afplay', filepath], check=True)
                return True
            
            elif system == 'Linux':
                # Try multiple commands
                for cmd in ['aplay', 'paplay', 'ffplay -nodisp -autoexit']:
                    if cls._command_exists(cmd.split()[0]):
                        subprocess.run(cmd.split() + [filepath], 
                                     stderr=subprocess.DEVNULL,
                                     check=True)
                        return True
                return False
            
            elif system == 'Windows':
                # Use PowerShell to play audio
                ps_script = f'''
                $sound = New-Object System.Media.SoundPlayer "{filepath}"
                $sound.PlaySync()
                '''
                subprocess.run(['powershell', '-Command', ps_script], check=True)
                return True
            
        except subprocess.CalledProcessError:
            return False
        
        return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        system = platform.system()
        
        if system == 'Darwin':
            return """
System audio (macOS):
  afplay - Should be pre-installed
  If not working, ensure system audio is configured correctly.
"""
        elif system == 'Linux':
            return """
System audio (Linux):
  Option 1: sudo apt-get install alsa-utils (provides aplay)
  Option 2: sudo apt-get install pulseaudio-utils (provides paplay)
  Option 3: sudo apt-get install ffmpeg (provides ffplay)
"""
        elif system == 'Windows':
            return """
System audio (Windows):
  PowerShell - Should be pre-installed
  If not working, ensure system audio is configured correctly.
"""
        else:
            return "System audio commands not available on this platform."


# List of backends in priority order
AUDIO_BACKENDS = [
    SoundDeviceBackend,
    PyAudioBackend,
    SimpleAudioBackend,
    SystemCommandBackend,
]


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def find_available_backend() -> Optional[type[AudioBackend]]:
    """Find the first available audio backend."""
    print("🔍 Probing for available audio backends...")
    
    for backend_class in AUDIO_BACKENDS:
        print(f"  Checking {backend_class.name}...", end=' ')
        if backend_class.is_available():
            print("✓ Available")
            return backend_class
        else:
            print("✗ Not available")
    
    return None


def print_installation_help():
    """Print detailed installation instructions for all backends."""
    print("\n" + "="*70)
    print("📦 INSTALLATION INSTRUCTIONS")
    print("="*70)
    print("\nNo audio backend is currently available.")
    print("Please install one of the following:\n")
    
    for i, backend_class in enumerate(AUDIO_BACKENDS, 1):
        print(f"{i}. {backend_class.name}")
        print(backend_class.get_install_instructions())
    
    print("="*70)
    print("\nAfter installation, run this script again.")


def play_instrument(instrument: Instrument, base_freq: float, 
                   backend: type[AudioBackend], duration: float = 5.0):
    """Play all strings of an instrument from highest to lowest."""
    
    print(f"\n🎸 Playing {instrument.name} strings (A4 = {base_freq} Hz)")
    print("="*70)
    
    for i, string in enumerate(instrument.get_all_strings(), 1):
        freq = string.calculate_frequency(base_freq)
        
        print(f"\n{i}. String {string} - {freq:.3f} Hz")
        print(f"   {'▶' * 40}")
        
        success = backend.play_tone(freq, duration)
        
        if not success:
            print(f"   ⚠️  Failed to play {string}")
            print(f"   Consider trying a different audio backend.")
        
        # Small pause between strings
        if i < len(instrument.strings):
            time.sleep(0.5)
    
    print("\n" + "="*70)
    print("✓ Tuning sequence complete!")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Guitar Tuner 432 Hz - Command Line Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --frequency 440
  %(prog)s --instrument guitalele
  %(prog)s --frequency 440 --instrument guitalele
  %(prog)s -f 440 -i guitalele

Available instruments: guitar, guitalele
        """
    )
    
    parser.add_argument(
        '-f', '--frequency',
        type=float,
        default=432.0,
        metavar='FREQ',
        help='Base frequency for A4 in Hz (default: 432.0, range: 400-460)'
    )
    
    parser.add_argument(
        '-i', '--instrument',
        type=str,
        default='guitar',
        choices=list(INSTRUMENTS.keys()),
        metavar='TYPE',
        help=f"Instrument type: {', '.join(INSTRUMENTS.keys())} (default: guitar)"
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=5.0,
        metavar='SECS',
        help='Duration of each tone in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '-l', '--list-backends',
        action='store_true',
        help='List all available audio backends and exit'
    )
    
    args = parser.parse_args()
    
    # Validate frequency
    if not 400 <= args.frequency <= 460:
        print(f"ERROR: Frequency must be between 400 and 460 Hz")
        print(f"You specified: {args.frequency} Hz")
        sys.exit(1)
    
    # Validate duration
    if args.duration <= 0:
        print(f"ERROR: Duration must be positive")
        print(f"You specified: {args.duration} seconds")
        sys.exit(1)
    
    # Check if numpy is available
    try:
        import numpy
    except ImportError:
        print("ERROR: numpy is required but not installed")
        print("\nInstallation:")
        print("  pip install numpy")
        sys.exit(1)
    
    # List backends if requested
    if args.list_backends:
        print("\n🎵 Audio Backends\n")
        for i, backend_class in enumerate(AUDIO_BACKENDS, 1):
            available = "✓ Available" if backend_class.is_available() else "✗ Not available"
            print(f"{i}. {backend_class.name:20} {available}")
        print()
        sys.exit(0)
    
    # Find available backend
    backend = find_available_backend()
    
    if backend is None:
        print("\n❌ ERROR: No audio backend is available!")
        print_installation_help()
        sys.exit(1)
    
    print(f"✓ Using {backend.name} backend\n")
    
    # Get instrument
    instrument = INSTRUMENTS[args.instrument]
    
    # Play tuning sequence
    try:
        play_instrument(instrument, args.frequency, backend, args.duration)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
