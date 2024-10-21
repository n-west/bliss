#Look for an executable called sphinx-build
find_program(SPHINX_BUILD_EXECUTABLE
             NAMES sphinx-build
             DOC "Path to sphinx-build executable")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
    REQUIRED_VARS
    SPHINX_BUILD_EXECUTABLE
)