# oracle ports - aborted launch

This directory records an aborted launch of the same scheme. The harness did
not set the case selector for the commit stage, so commit failed with
`invalid_config`, prove and verify were blocked, and no proof was produced. It
is kept as the failure record; the measurements are in the completed trace
directory next to it.
