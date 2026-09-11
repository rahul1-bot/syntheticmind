# syntheticmind.storage: Artifact Storage Boundaries

`storage/` owns filesystem protocol resolution, local and remote path semantics, text and binary resources, artifact staging, copying, removal, metadata queries, and directory operations shared by checkpoint, logger, profiler, and Trainer persistence.

`StoragePath` is one path-bound external-resource owner. Construction resolves an fsspec backend and backend-local path once; instance operations then perform normalization, joining, containment, metadata, resource opening, directory traversal, mutation, and local staging without a module-function or static utility surface. Public training, callback, logger, and profiler APIs continue accepting ordinary `str | Path` values.

`filesystem.py` is the single current storage boundary. Additional storage classes or modules require a distinct backend lifecycle or artifact contract rather than a desire to shorten this implementation.

| Module | Contents |
|---|---|
| `filesystem.py` | `StoragePath`, one resolved local or fsspec-backed resource identity. |

## Resource model

Construction resolves the fsspec backend and backend-local path without touching the resource. Local normalization expands the user directory and applies operating-system semantics. Remote identities retain their protocol and use POSIX joining.

```python
from syntheticmind.storage.filesystem import StoragePath

artifact: StoragePath = StoragePath("memory://experiments/run-7/metrics.csv")
StoragePath(artifact.parent).make_directory()
artifact.write_text("step,loss\n0,1.0\n")
```

Framework clients usually continue passing `str | Path`; checkpoint, logger, profiler, and Trainer owners construct `StoragePath` at their external boundary. Direct use is appropriate for a custom persistence component that genuinely owns a filesystem resource.

## Local staging

Some external libraries can write only local files. `local_artifact` yields the direct local destination or a temporary local path for a remote resource. When the remote context exits successfully and a file exists, the completed artifact is uploaded.

```python
with StoragePath(output_path).local_artifact() as local_path:
    external_writer.export(local_path)
```

The context does not decide artifact naming, atomicity, retention, or rank ownership. The calling semantic component retains those policies.

## Safety boundaries

`same_as` compares normalized resource identities. `is_within` performs strict descendant testing and supports checkpoint path-escape protection. Recursive removal is never implicit; the caller must request it and remains responsible for proving semantic ownership of the target.

Remote append reconstructs and replaces the complete text value because generic fsspec backends do not share dependable append behavior. Components requiring transaction guarantees must use their backend transaction explicitly, as checkpoint and CSV persistence do.

## Related components

`../plugins/checkpoint_io.py`, `../callbacks/checkpointing/`, `../loggers/`, and `../profilers/` retain artifact semantics while consuming this resource boundary.
