def pytest_addoption(parser):
    parser.addoption("--dataroot", action="store", type=str)
    parser.addoption("--host", action="store", type=str)
    parser.addoption("--port", action="store", type=int)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    dataroot_value = metafunc.config.option.dataroot
    if "dataroot" in metafunc.fixturenames:
        metafunc.parametrize("dataroot", [dataroot_value])

    host_value = metafunc.config.option.host
    if "host" in metafunc.fixturenames:
        metafunc.parametrize("host", [host_value])

    port_value = metafunc.config.option.port
    if "port" in metafunc.fixturenames:
        metafunc.parametrize("port", [port_value])
