"""GET /sessions/{id}/entities: every participant with its staged dbId, the reactions using it (with roles) and the
complexes/sets containing it; private like every session route."""
from tests.test_edit_api import H, env  # noqa: F401  (pytest fixture)


def test_entities_list_participants_with_their_staged_dbids_and_uses(env):
    c, sid, _ = env
    r = c.get(f'/api/llm/sessions/{sid}/entities', headers=H())
    assert r.status_code == 200
    by = {e['key']: e for e in r.json()}
    assert set(by) >= {'pink1', 'ub', 'atp', 'cccp'}
    assert by['pink1']['participant']['name'] == 'PINK1' and by['pink1']['reactions'] == [
        {'key': 'r0', 'name': 'PINK1 phosphorylates ubiquitin', 'roles': ['catalyst']}]
    assert by['ub']['reactions'][0]['roles'] == ['input', 'output']
    assert by['cccp']['reactions'] == [] and by['cccp']['partOf'] == []
    staged = {i['dbId']: i for i in c.get(f'/api/llm/sessions/{sid}/export', headers=H()).json()['newInstances']}
    assert staged[by['pink1']['dbId']]['displayName'].startswith('PINK1')        # the dbId is the exported instance's


def test_entities_are_private_and_an_old_session_still_gets_dbids(env):
    c, sid, _ = env
    assert c.get(f'/api/llm/sessions/{sid}/entities', headers=H('bob')).status_code == 404
    svc = c.app.state.service
    s = svc.store.get(sid, 'alice')
    s.participant_to_db_id = {}                       # saved before the mapping was kept
    svc.store.save(s)
    by = {e['key']: e for e in c.get(f'/api/llm/sessions/{sid}/entities', headers=H()).json()}
    assert by['pink1']['dbId'] is not None and by['pink1']['dbId'] < 0
