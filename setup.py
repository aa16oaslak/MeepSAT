
# setup.py
"""
Setup script for MEEPSAT package with post-installation instructions.
"""

from setuptools import setup
from setuptools.command.install import install
import sys

class PostInstallCommand(install):
    """Post-installation message."""
    def run(self):
        install.run(self)
        print("\n" + "="*70)
        print("MEEPSAT Installation Complete!")
        print("="*70)
        print("\n⚠️  IMPORTANT: FFmpeg is required for animation generation\n")
        print("Please ensure FFmpeg is installed on your system:\n")
        print("Ubuntu/Debian:")
        print("  sudo apt-get install ffmpeg\n")
        print("macOS:")
        print("  brew install ffmpeg\n")
        print("Windows:")
        print("  Download from https://ffmpeg.org/download.html")
        print("\n" + "="*70 + "\n")

setup(
    cmdclass={
        'install': PostInstallCommand,
    }
)