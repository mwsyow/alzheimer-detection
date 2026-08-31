"""Shared stand-ins for the wandb objects the training and evaluation paths log to.

Kept here rather than in one test module because both the CV reporting tests and the
training-loop tests need a logger that records instead of uploading.
"""


class FakeSummary(dict):
    """A wandb summary is a MutableMapping whose deletions reach the server."""


class FakeRun:
    """Records what would have been logged, so tests can assert on the exact key set."""

    def __init__(self, summary=None, name="fake-run", run_id="fake0000"):
        self.summary = FakeSummary(summary or {})
        self.name = name
        self.id = run_id
        self.sweep_id = None
        # Every log() call in order, so a test can look at one epoch rather than the
        # union of all of them -- a key logged once and then dropped is a real bug.
        self.log_calls = []
        self.logged = {}
        self.finished = False

    def log(self, values):
        self.log_calls.append(dict(values))
        self.logged.update(values)

    def define_metric(self, *args, **kwargs):
        pass

    def finish(self):
        self.finished = True
