import manimx


def test_package_imports() -> None:
    assert manimx.__version__ == "0.1.0"
    assert manimx.package_name() == "manimx"
