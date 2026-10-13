def test_packages_import():
    import roi_alert.cfg  # noqa: F401
    import roi_alert.dwell  # noqa: F401
    import roi_alert.events  # noqa: F401
    import roi_alert.evidence  # noqa: F401
    import roi_alert.runner  # noqa: F401
    import roi_alert.tracker  # noqa: F401
    import roi_alert.vision  # noqa: F401
    import webapp.main  # noqa: F401


def test_heavy_backend_is_stubbed():
    import ultralytics  # noqa: F401
