
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_app():
    for key in list(sys.modules.keys()):
        if key.startswith("app"):
            del sys.modules[key]

    with patch("streamlit.set_page_config"), \
         patch("streamlit.cache_resource", lambda **kw: (lambda f: f)):
        import app.main as app_module
    return app_module

class TestModuleImportable:
    def test_import_succeeds(self):
        app = _import_app()
        assert app is not None

    def test_main_callable(self):
        app = _import_app()
        assert callable(app.main)

    def test_load_detector_callable(self):
        app = _import_app()
        assert callable(app._load_detector)


class TestConstants:
    def setup_method(self):
        self.app = _import_app()

    def test_tab_config_has_four_entries(self):
        assert len(self.app._TAB_CONFIG) == 4

    def test_tab_config_slugs_are_strings(self):
        for label, itype, slug in self.app._TAB_CONFIG:
            assert isinstance(slug, str)

    def test_tab_config_covers_all_input_types(self):
        from src.detector import InputType
        config_types = {itype for _, itype, _ in self.app._TAB_CONFIG}
        assert config_types == set(InputType)

    def test_placeholder_covers_all_slugs(self):
        slugs = {slug for _, _, slug in self.app._TAB_CONFIG}
        assert slugs == set(self.app._PLACEHOLDER.keys())

    def test_help_covers_all_slugs(self):
        slugs = {slug for _, _, slug in self.app._TAB_CONFIG}
        assert slugs == set(self.app._HELP.keys())

    def test_model_dir_is_path(self):
        assert isinstance(self.app._MODEL_DIR, Path)

    def test_model_dir_points_to_models_folder(self):
        assert self.app._MODEL_DIR.name == "models"

class TestLoadDetector:
    def test_returns_detector_when_models_exist(self):
        app = _import_app()
        model_dir = _ROOT / "models"
        if not (model_dir / "url_model.joblib").exists():
            pytest.skip("Models not trained; skipping detector load test.")
        result = app._load_detector()
        from src.detector import Detector
        assert isinstance(result, Detector)

    def test_returns_none_when_model_dir_missing(self, tmp_path, monkeypatch):
        app = _import_app()
        monkeypatch.setattr(app, "_MODEL_DIR", tmp_path)
        result = app._load_detector()
        assert result is None

    def test_returns_none_when_models_absent(self, tmp_path, monkeypatch):
        app = _import_app()
        monkeypatch.setattr(app, "_MODEL_DIR", tmp_path)
        result = app._load_detector()
        assert result is None
