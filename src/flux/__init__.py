# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team.
# Original file was released at https://github.com/XLabs-AI/x-flux

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from ._version import version as __version__  # type: ignore
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent
