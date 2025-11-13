<!-- 39ca5d3c-aba3-4714-97e3-1daec556568a afbbbb7f-d335-4334-a4f0-e938c19018c5 -->
# Share Unit Aliases Across Package

1. **Introduce shared alias utilities** ✓

- File: `pypillometry/convenience.py`  
- Add a module-level `UNIT_ALIASES` dict (e.g., mapping "sec", "seconds", "s" → "sec") and a `normalize_unit(unit, *, allow_none, default)` helper that lower-cases, strips whitespace, checks membership, and returns canonical names (handling None/default cases).  
- Create/extend unit utility tests (e.g., new `tests/test_convenience.py::TestNormalizeUnit`) covering canonical returns, alias expansion, None handling, and error raising.  
- After this change, run: `conda run -n pypil pytest tests/test_convenience.py`.

2. **Refactor `GenericEyeData._unit_fac` to use aliases** ✓

- File: `pypillometry/eyedata/generic.py`  
- Replace manual `if` ladder with `normalize_unit` calls (falling back to canonical ms factor) so any alias resolves before computing factors.  
- Update/extend existing tests that exercise `_unit_fac` indirectly (e.g., `TestEyeData.test_get_duration`) with new alias cases.  
- Run: `conda run -n pypil pytest tests/test_eyedata.py::TestEyeData::test_get_duration`.

3. **Align time accessor with shared helper** ✓

- File: `pypillometry/eyedata/generic.py` (`_get_time_array`)  
- Remove inline alias dict, call `normalize_unit` (allowing None) and compute factors via `_unit_fac`; ensure unsupported aliases raise consistently.  
- Extend accessor tests (e.g., add parametrized alias assertions in `test_dunder_getitem_time_accessors`).  
- Run: `conda run -n pypil pytest tests/test_eyedata.py::TestEyeData::test_dunder_getitem_time_accessors`.

4. **Normalize units in blink helpers** ✓

- File: `pypillometry/eyedata/generic.py` (`get_blinks`)  
- Normalize requested `units` before comparison (`None` vs canonical), ensuring alias inputs ("seconds", "hrs", etc.) propagate to `Intervals.to_units`.  
- Update blink-related tests (e.g., `test_get_blinks_with_units`, `test_blinks_merge`) to cover alias inputs.  
- Run: `conda run -n pypil pytest tests/test_eyedata.py::TestEyeData::test_get_blinks_with_units tests/test_eyedata.py::TestEyeData::test_blinks_merge`.

5. **Normalize units in event accessors** ✓

- File: `pypillometry/eyedata/generic.py` (`get_events`, `set_events`)  
- Use `normalize_unit` for incoming/outgoing units and ensure conversions target canonical names before delegating to `Events`.  
- Add/adjust tests around event retrieval (`test_get_events` cases) to include alias coverage.  
- Run: `conda run -n pypil pytest tests/test_eyedata.py::TestEventsIntegration::test_get_events_with_different_units`.

6. **Update `Intervals` conversions**  

- File: `pypillometry/intervals.py` (`Intervals.as_index`, `Intervals.to_units`)  
- Replace local `units_to_ms` dicts with calls to `normalize_unit`, support aliases for both source and target; update docstrings accordingly.  
- Extend interval tests in `tests/test_intervals_class.py` to exercise alias conversions (e.g., `secs`, `minutes`).  
- Run: `conda run -n pypil pytest tests/test_intervals_class.py`.

7. **Update `Events` conversions**  

- File: `pypillometry/events.py` (`Events.__init__`, `Events.to_units`, `Events.plot`)  
- Normalize stored units in `__init__`, reuse helper in `to_units`, and ensure plotting/default behaviour handles aliases.  
- Extend `tests/test_events.py` for alias inputs/outputs and plotting conversions as needed.  
- Run: `conda run -n pypil pytest tests/test_events.py`.

8. **Final verification**  

- Sweep for any remaining `units` handling (e.g., plotting modules already calling `_unit_fac`) to ensure they indirectly benefit or add normalization if direct strings remain.  
- Execute a broader regression suite focusing on eyedata/intervals/events: `conda run -n pypil pytest tests/test_eyedata.py tests/test_intervals_class.py tests/test_events.py`.

