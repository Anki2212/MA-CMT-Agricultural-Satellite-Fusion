"""
Integration tests for FastAPI endpoints with trained checkpoints and experiment results.
Can be executed directly or with: python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.main import (
    root,
    health,
    dataset_info,
    list_samples,
    get_sample,
    predict,
    PredictionRequest,
    models_status,
    experiment_results,
    experiment_summary,
    get_confusion_matrix,
    get_experiment_plot,
)
from ml.dataset import CLASS_NAMES


def test_root_endpoint():
    res = root()
    assert res["status"] == "ok"
    assert "MA-CMT" in res["message"]


def test_health_endpoint():
    res = health()
    assert res["status"] == "ok"
    assert "device" in res
    assert res["data_cache_exists"] is True
    assert res["experiments_ready"] is True
    # All 4 checkpoints must be ready
    assert all(res["checkpoints"].values())


def test_dataset_info_endpoint():
    res = dataset_info()
    assert res["total_samples"] == 42603
    assert res["n_classes"] == 10
    assert res["n_months"] == 12
    assert res["n_features"] == 13
    assert len(res["class_names"]) == 10


def test_list_samples_endpoint():
    res = list_samples(split="test", class_filter=None, limit=10, offset=0)
    assert res["total"] > 0
    assert len(res["samples"]) == 10
    sample = res["samples"][0]
    assert "sample_index" in sample
    assert "label" in sample
    assert "id" in sample


def test_get_sample_detail_endpoint():
    res = get_sample(0)
    assert res["sample_index"] == 0
    assert "label" in res
    assert len(res["monthly_observations"]) == 12
    first_month = res["monthly_observations"][0]
    assert "s1" in first_month and "s2" in first_month
    assert "VV" in first_month["s1"] and "VH" in first_month["s1"]


def test_get_sample_invalid_index():
    with pytest.raises(HTTPException) as exc:
        get_sample(999999)
    assert exc.value.status_code == 404


def test_models_status_all_available():
    res = models_status()
    for m in ["macmt", "s1only", "s2only", "concat"]:
        assert m in res
        assert res[m]["available"] is True
        assert "epoch" in res[m]
        assert "val_acc" in res[m]
        assert res[m]["val_acc"] > 0.3


def test_predict_real_checkpoint_macmt():
    req = PredictionRequest(
        sample_index=5,
        model_name="macmt",
        missing_months_s1=[],
        missing_months_s2=[],
    )
    res = predict(req)
    assert res["model"] == "macmt"
    assert res["predicted_class"] in CLASS_NAMES
    assert 0 <= res["predicted_index"] < 10
    assert res["true_class"] in CLASS_NAMES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 10
    # Probabilities should sum to approximately 1.0
    assert abs(sum(res["probabilities"].values()) - 1.0) < 0.01


def test_predict_with_missing_masks():
    req = PredictionRequest(
        sample_index=5,
        model_name="macmt",
        missing_months_s1=[0, 1, 2, 3, 4, 5],
        missing_months_s2=[6, 7, 8],
    )
    res = predict(req)
    assert res["model"] == "macmt"
    assert res["predicted_class"] in CLASS_NAMES
    assert res["missing_months_s1"] == [0, 1, 2, 3, 4, 5]
    assert res["missing_months_s2"] == [6, 7, 8]


def test_predict_baselines():
    for baseline in ["s1only", "s2only", "concat"]:
        req = PredictionRequest(
            sample_index=5,
            model_name=baseline,
            missing_months_s1=[],
            missing_months_s2=[],
        )
        res = predict(req)
        assert res["model"] == baseline
        assert res["predicted_class"] in CLASS_NAMES


def test_experiment_results_available():
    res = experiment_results()
    for scenario in ["complete", "random", "s1_missing", "s2_missing", "consecutive"]:
        assert scenario in res


def test_experiment_summary():
    summ = experiment_summary()
    assert "summary" in summ
    assert len(summ["summary"]) >= 80
    row = summ["summary"][0]
    assert "scenario" in row
    assert "model" in row
    assert "accuracy" in row
    assert "f1_macro" in row


def test_confusion_matrix():
    cm = get_confusion_matrix(model="macmt", scenario="complete", missing_rate=0)
    assert cm["model"] == "macmt"
    assert len(cm["confusion_matrix"]) == 10
    assert len(cm["confusion_matrix"][0]) == 10
    assert len(cm["class_names"]) == 10


def test_experiment_plot_endpoint():
    res = get_experiment_plot()
    assert res.media_type == "image/png"
    assert Path(res.path).exists()
