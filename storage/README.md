# syntheticmind.storage: Artifact Storage Boundaries

`storage/` owns filesystem protocol resolution, local and remote path semantics, text and binary resources, artifact staging, copying, removal, metadata queries, and directory operations shared by checkpoint, logger, profiler, and Trainer persistence.

`StoragePath` is one path-bound external-resource owner. Construction resolves an fsspec backend and backend-local path once; instance operations then perform normalization, joining, containment, metadata, resource opening, directory traversal, mutation, and local staging without a module-function or static utility surface. Public training, callback, logger, and profiler APIs continue accepting ordinary `str | Path` values.

`filesystem.py` is the single current storage boundary. Additional storage classes or modules require a distinct backend lifecycle or artifact contract rather than a desire to shorten this implementation.
