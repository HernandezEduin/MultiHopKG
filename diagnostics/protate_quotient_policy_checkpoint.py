"""Run the existing checkpoint diagnostic with quotient-space pRotatE enabled."""

from __future__ import annotations

import diagnostics.protate_policy_checkpoint as base
from temporary_patches.protate_navigation import enable_protate_navigation_patches
from temporary_patches.protate_policy import enable_protate_policy_patch
from temporary_patches.protate_quotient import enable_protate_quotient_navigation_patch


if __name__ == "__main__":
    enable_protate_navigation_patches()
    enable_protate_quotient_navigation_patch()
    enable_protate_policy_patch()
    base.main()
