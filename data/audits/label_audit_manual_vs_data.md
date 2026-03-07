# Label Audit: Dataset vs CHAT Manual

- Labels in dataset: **42**
- Labels matched in manual text: **37**
- Labels not found in manual text: **5**

## Labels Not Found (needs manual check)
- `[* s:r:gc:det]` (count=199)
- `[* m:++er]` (count=153)
- `[* m:0est]` (count=150)
- `[* m:0er]` (count=146)
- `[* m:++est]` (count=144)

## Notes
- Match is text-based against extracted PDF content; OCR/extraction noise can cause false negatives.
- For robust decisions, verify all `not found` labels directly in the manual section on error coding.