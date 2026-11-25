"""Probability references must use fitting labels, never test/calibration outcomes."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest
from overfit_radar.config import get_project_root


def test_probability_references_and_analytic_brier():
    spec=importlib.util.spec_from_file_location('experiment',get_project_root()/'scripts/phase6_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fit=pd.DataFrame({'survived':[0,1,1,1]})
    references=module.probability_references(fit,index=pd.Index([10,20]))
    assert list(references.index)==[10,20]
    assert list(references.constant_05)==[.5,.5]
    assert list(references.fit_prevalence)==[.75,.75]
    assert module.probability_metrics([0,1],references.constant_05)['brier']==pytest.approx(.25)
