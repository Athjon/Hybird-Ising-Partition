"""Parser checks for the reproducible real-hypergraph benchmark."""

import pytest

from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr


def test_read_unweighted_hgr_preserves_declared_isolated_vertices(tmp_path):
    path = tmp_path / 'case.hgr'
    path.write_text('% example\n2 5\n1 3 4\n2 3\n', encoding='utf-8')
    assert read_unweighted_hgr(path) == (5, [[0, 2, 3], [1, 2]])


@pytest.mark.parametrize('content,match', [
    ('1 3 1\n2 1 2\n', 'unweighted'),
    ('2 3\n1 2\n', 'header says'),
    ('1 3\n1 4\n', 'out-of-range'),
])
def test_read_unweighted_hgr_rejects_bad_inputs(tmp_path, content, match):
    path = tmp_path / 'bad.hgr'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        read_unweighted_hgr(path)
