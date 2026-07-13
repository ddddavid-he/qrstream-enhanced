"""Compatibility constraints for runtime dependencies."""

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def _project_requirement(name: str) -> Requirement:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        candidate = raw_line.strip().rstrip(",").strip('"')
        requirement = Requirement(candidate) if candidate.startswith(name) else None
        if requirement is not None and requirement.name == name:
            return requirement
    raise AssertionError(f"project dependency {name!r} was not found")


def test_zxing_cpp_excludes_known_real_world_regression() -> None:
    requirement = _project_requirement("zxing-cpp")

    assert Version("3.0.0") in requirement.specifier
    assert Version("3.1.0") not in requirement.specifier
