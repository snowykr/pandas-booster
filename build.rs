fn main() {
    if std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_none() {
        pyo3_build_config::add_libpython_rpath_link_args();
    }
}
