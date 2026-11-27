All the files have been copy/pasted from
https://github.com/NVIDIA/stdexec/pull/1790/changes#diff-508a8f945cb88dc3abec0b3739ef3f2b371b416caf5b472dff484cc63adc758c.

That pull request builds its senders with `stdexec::__sexpr`, which https://github.com/NVIDIA/stdexec/pull/2531
removed. Here, `enter_scopes`, `invoke`, `lifetime` and `within` build a `stdexec::__facade::__sender` instead, and
their implementations derive from `stdexec::__facade::__defaults`. Nothing else is ported: each algorithm still gives
its completion signatures and attributes itself, and connects its children in its own operation state.
