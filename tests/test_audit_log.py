from node.audit_log import AuditEventType, AuditLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_log(*events) -> AuditLog:
    """Build an AuditLog with the given (event_type, actor, payload) tuples."""
    log = AuditLog()
    for event_type, actor, payload in events:
        log.append(event_type, actor, payload)
    return log


def _three_entry_log() -> AuditLog:
    return make_log(
        (AuditEventType.NODE_REGISTERED, "node-A", {"stake": 100}),
        (AuditEventType.JOB_CLAIMED, "node-A", {"job_id": "j1"}),
        (AuditEventType.JOB_COMPLETED, "node-A", {"job_id": "j1", "tokens": 512}),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_append_increments_sequence():
    log = AuditLog()
    e0 = log.append(AuditEventType.NODE_REGISTERED, "node-1", {})
    e1 = log.append(AuditEventType.JOB_CLAIMED, "node-1", {"job_id": "x"})
    e2 = log.append(AuditEventType.JOB_COMPLETED, "node-1", {"job_id": "x"})
    assert e0.sequence == 0
    assert e1.sequence == 1
    assert e2.sequence == 2


def test_entry_hash_computed_on_creation():
    log = AuditLog()
    entry = log.append(AuditEventType.NODE_REGISTERED, "node-1", {})
    assert entry.entry_hash != ""
    assert len(entry.entry_hash) == 64  # SHA-256 hex digest


def test_entry_verify_passes():
    log = AuditLog()
    entry = log.append(AuditEventType.PAYMENT_RECEIVED, "node-1", {"amount": 50})
    assert entry.verify() is True


def test_entry_verify_fails_after_tamper():
    log = AuditLog()
    entry = log.append(AuditEventType.PAYMENT_RECEIVED, "node-1", {"amount": 50})
    # Tamper with payload after creation
    entry.payload["amount"] = 9999
    assert entry.verify() is False


def test_chain_verify_valid():
    log = _three_entry_log()
    assert log.verify_chain() is True


def test_chain_verify_fails_if_hash_altered():
    log = _three_entry_log()
    # Alter the stored hash of the middle entry
    log[1].entry_hash = "deadbeef" * 8
    assert log.verify_chain() is False


def test_chain_verify_fails_if_prev_broken():
    log = _three_entry_log()
    # Break the prev_hash link on entry 2
    log[2].prev_hash = "00" * 32
    # Also recompute entry_hash so the individual verify() passes but chain link is wrong
    log[2].entry_hash = log[2]._compute_hash()
    assert log.verify_chain() is False


def test_entries_by_type():
    log = make_log(
        (AuditEventType.NODE_REGISTERED, "node-A", {}),
        (AuditEventType.JOB_CLAIMED, "node-A", {"job_id": "j1"}),
        (AuditEventType.JOB_CLAIMED, "node-B", {"job_id": "j2"}),
        (AuditEventType.JOB_COMPLETED, "node-A", {"job_id": "j1"}),
    )
    claimed = log.entries_by_type(AuditEventType.JOB_CLAIMED)
    assert len(claimed) == 2
    assert all(e.event_type == AuditEventType.JOB_CLAIMED for e in claimed)


def test_entries_by_actor():
    log = make_log(
        (AuditEventType.NODE_REGISTERED, "node-A", {}),
        (AuditEventType.NODE_REGISTERED, "node-B", {}),
        (AuditEventType.JOB_CLAIMED, "node-A", {"job_id": "j1"}),
        (AuditEventType.JOB_CLAIMED, "node-B", {"job_id": "j2"}),
    )
    node_a_entries = log.entries_by_actor("node-A")
    assert len(node_a_entries) == 2
    assert all(e.actor == "node-A" for e in node_a_entries)


def test_len():
    log = _three_entry_log()
    assert len(log) == 3


def test_getitem():
    log = _three_entry_log()
    first = log[0]
    assert first.sequence == 0
    assert first.event_type == AuditEventType.NODE_REGISTERED


def test_first_entry_has_empty_prev_hash():
    log = AuditLog()
    entry = log.append(AuditEventType.NODE_REGISTERED, "node-1", {})
    assert entry.prev_hash == ""


def test_each_entry_links_to_previous():
    log = _three_entry_log()
    assert log[1].prev_hash == log[0].entry_hash
    assert log[2].prev_hash == log[1].entry_hash
