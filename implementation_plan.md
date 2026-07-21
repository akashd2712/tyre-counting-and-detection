# Bidirectional Counting Logic Implementation Plan

Provide a brief description of the problem, any background context, and what the change accomplishes.
Currently, the `TyreCounter` class in `counting/counter.py` only tracks crossings in a single direction defined by the `entry_direction` configuration. It ignores any objects crossing the line in the opposite direction. To align the code with the Product Requirements Document (PRD), we need to update the logic so that it simultaneously detects both entry and exit crossings, upcounting the inventory when a tyre enters the container and downcounting when it rolls out.

## User Review Required
This is a straightforward behavioral fix to align the code with the stated PRD requirements. No major architectural changes are needed.

## Open Questions
None. The requested logic is well-defined by the PRD's container loading and unloading flow.

## Proposed Changes

### counting

#### [MODIFY] [counter.py](file:///a:/tyre/counting/counter.py)
Update the `evaluate_crossing` method:
- Remove the single `crossed` boolean logic that checks only one direction.
- Add logic to explicitly detect `crossed_down` and `crossed_up`.
- Adjust inventory and event counts based on whether the crossing direction matches the `entry_direction` or opposes it.
  - If `crossed_down` and `entry_direction == "down"`, then increment entry and inventory.
  - If `crossed_up` and `entry_direction == "down"`, then increment exit and decrement inventory.
  - Ensure the inventory `current_count` never drops below 0.

## Verification Plan

### Manual Verification
- Run the main pipeline on a sample video containing tyres moving in both directions.
- Observe the on-screen display.
- Verify that a tyre crossing downward increments the Total Count and Entry Count.
- Verify that a tyre crossing upward decrements the Total Count and increments the Exit Count.
