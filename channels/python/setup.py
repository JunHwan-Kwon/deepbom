from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from os import environ
from re import fullmatch
from sysconfig import get_platform


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        platform = environ.get("DEEPBOM_WHEEL_PLATFORM_TAG", "").strip()
        if platform and not fullmatch(r"manylinux_2_28_(?:x86_64|aarch64)", platform):
            raise RuntimeError("DEEPBOM_WHEEL_PLATFORM_TAG must be a supported manylinux_2_28 tag")
        if not platform:
            platform = get_platform().replace("-", "_").replace(".", "_")
        return "py3", "none", platform


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel})
