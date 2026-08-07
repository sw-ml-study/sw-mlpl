# Saga: partial-hof-adoption
demo-combinators agent found two consumers that never adopted
the shared apply_callable path (the combinator-birds design
CLAIMED they did): the Result combinators (map_ok/and_then/
or_else) and bracket's use/teardown hooks both rejected
Value::Partial. Fix: route UserFnRef|Partial through
callable_apply::apply_callable; keep builtin restrictions;
setup stays a raw zero-arg reference. One step + close.
