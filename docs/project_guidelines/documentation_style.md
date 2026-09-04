# SyntheticMind Documentation Style

SyntheticMind source documentation follows the line-comment style established by the completed VOCODE harness. Rahul explicitly authorized the dedicated documentation phase on 04 September 2026 after the private core runtime stabilized. This project-level decision supersedes the generic no-comment default for SyntheticMind only.

## Source format

Every Python module begins with this structure:

```python
# This module:
# 1. Defines the principal concept and its complete responsibility
# 2. Names additional colocated concepts only when they are genuinely owned here
#
# Design decisions:
# - Explains the invariants, lifecycle, dependency direction, or failure policy
# - Explains why the implementation takes this form rather than narrating syntax
#
# Author: Rahul Sawhney
```

Triple-quoted docstrings are not used. Documentation remains ordinary `#` comments so the source matches the established project style exactly.

## Class documentation

A public or architecturally significant class receives a comment block immediately inside the class. The block explains:

1. What state or lifecycle the class owns.
2. Which component constructs and consumes it.
3. What a downstream client subclasses, configures, or calls.
4. Which neighboring concepts remain outside its authority.
5. A realistic client example when the class is an extension surface.

Private support classes receive a shorter ownership statement. Their comments must explain why the class exists independently rather than merely restating its name.

## Method documentation

Public constructors, extension hooks, and client-facing operations use comment-form sections when relevant:

```python
def operation(self, value: Value) -> Result:
    # Performs the lifecycle operation and states its observable contract.
    #
    # Args:
    #     value: Meaning, accepted forms, and important defaults.
    #
    # Return:
    #     Observable result and ownership semantics.
    #
    # Raises:
    #     SpecificError: Exact invalid condition and corrective boundary.
```

Internal methods are documented when their ordering, rollback, concurrency, persistence, distributed participation, or numerical behavior is not inferable from the name and type annotations alone.

## Client integration

Comments must teach downstream researchers how their code enters the harness. Integration documentation covers the relevant subclass, overridden methods, hook timing, accepted return values, configuration ownership, checkpoint behavior, and distributed constraints. Examples use current canonical imports and executable API shapes.

The following are prohibited:

1. Examples copied from an obsolete SyntheticMind version.
2. Lightning-branded client identifiers or vendor-specific product language.
3. Descriptions of unsupported capabilities.
4. Generic examples that bypass Module, DataModule, Trainer, Strategy, or other established extension contracts.

## Quality rules

A comment must add information unavailable from the adjacent syntax. Useful comment subjects include:

1. Lifecycle order and hook timing.
2. State ownership and checkpoint participation.
3. Distributed collective participation and rank ownership.
4. Exact continuation and transaction boundaries.
5. Numerical precision, gradient synchronization, and optimizer cadence.
6. Failure rollback and cleanup guarantees.
7. Client extension and substitution contracts.

Comments that merely translate code into English are rejected. Examples include “increments the counter,” “returns the value,” or “loops through the items” when no additional invariant is explained.

## Maintenance rules

Comments and Markdown documentation are part of the reviewed contract after this phase. Any code change that invalidates documented behavior must update the corresponding prose in the same change. Future agents may not delete, rewrite, or relocate documentation merely to satisfy a generic no-comment instruction; changes require an actual semantic correction or Rahul's explicit direction.

No `TODO` comments are permitted. Unfinished work belongs in `progress.md` with a concrete acceptance boundary. Pydantic field descriptions remain unnecessary because model field names, types, constraints, and defaults already provide machine-readable schema metadata.

## Markdown documentation

Every semantic package carries a `README.md` covering ownership, module inventory, client integration, important runtime contracts, and related components. Cross-package architecture and guides live under semantically named directories in `docs/`. Ledgers remain decision and progress indexes rather than duplicated API manuals.

## Review checklist

1. Every module has a current ownership header and Rahul author line.
2. Every public extension class explains client integration.
3. Every public hook states lifecycle timing and default behavior.
4. Every complex persistence, distributed, or rollback transition explains its invariant.
5. Every example uses canonical current imports and signatures.
6. No comment claims behavior that lacks executable evidence.
7. No docstring, TODO, decorative banner, or syntax narration exists.
8. Package READMEs and root documentation agree with the source topology.
