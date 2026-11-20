import tuncall


def test_package_imports() -> None:
    assert isinstance(tuncall.__version__, str)
