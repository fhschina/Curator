# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Limit the optional specialist's authority without changing its observed answers."""

from copy import deepcopy

from eval.dedup.judging import critic_subject_binding as previous

CONTRACT = "dedup-critic-subject-scope-v2"
VETO_BINDINGS = frozenset(("LIABILITY_PARTY", "POLICY_SERVICE", "FAILED_OBJECT"))
response_schema = previous.response_schema
messages = previous.messages
route = previous.route


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = route(main, payload)
    if routing != "REVIEW_BILATERAL_SUBJECTS":
        return deepcopy(main), routing
    proof = previous.compile_review(value, payload)
    if proof is None:
        return deepcopy(main), "NO_SUPPORTED_SUBJECT_VETO_KEEP_COVERAGE"
    if value["binding_type"] not in VETO_BINDINGS:
        return deepcopy(main), "SUBJECT_OBJECTION_OUTSIDE_SPECIALIST_AUTHORITY"
    return previous.veto.apply_review(main, payload, proof)
